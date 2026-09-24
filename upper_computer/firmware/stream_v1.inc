/* Included by main.c after peripheral declarations. No D-cache is enabled. */
#define ADC_BUF_LEN 40U
#define HALF_SAMPLES 20U
#define TX_SLOTS 32U
#define FRAME_MAX 96U

/* DMA1 cannot access DTCM. stream.sct places these in SRAM D2 explicitly. */
static uint16_t adc_buffer[ADC_BUF_LEN] __attribute__((section(".RAM_D2"), aligned(32)));
static uint8_t tx_data[TX_SLOTS][FRAME_MAX] __attribute__((section(".RAM_D2"), aligned(32)));
static uint16_t tx_len[TX_SLOTS];
static volatile uint32_t tx_head, tx_tail, frame_seq, samples_completed, dropped;
static volatile uint16_t stream_faults;
static volatile uint8_t tx_busy;
static uint8_t rx_byte;
static char command_line[32];
static uint8_t command_len;
static uint8_t command_overflow;

static void put16(uint8_t *p, uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void put32(uint8_t *p, uint32_t v) {
  put16(p,(uint16_t)v); put16(p+2,(uint16_t)(v>>16));
}
static uint16_t crc16(const uint8_t *p, uint16_t n) {
  uint16_t crc=0xFFFFU;
  for (uint16_t i=0;i<n;i++) {
    crc ^= (uint16_t)p[i]<<8;
    for (uint8_t j=0;j<8;j++) crc=(crc&0x8000U)?(uint16_t)((crc<<1)^0x1021U):(uint16_t)(crc<<1);
  }
  return crc;
}
static void enqueue_frame(uint8_t type, uint32_t first, const uint8_t *payload, uint16_t len, uint16_t count) {
  uint32_t saved=__get_PRIMASK(); __disable_irq();
  uint32_t seq=frame_seq++;
  if ((uint32_t)(tx_head-tx_tail)>=TX_SLOTS) {
    dropped++; stream_faults|=1U; __set_PRIMASK(saved); return;
  }
  uint32_t slot=tx_head%TX_SLOTS;
  uint8_t *p=tx_data[slot];
  p[0]=0xA5U; p[1]=0x5AU; p[2]=1U; p[3]=type;
  put16(p+4,len); put32(p+6,seq); put32(p+10,first);
  put16(p+14,count); put16(p+16,stream_faults);
  memcpy(p+18,payload,len);
  put16(p+18+len,crc16(p,(uint16_t)(18+len)));
  tx_len[slot]=(uint16_t)(20+len);
  __DMB(); tx_head++;
  __set_PRIMASK(saved);
}
static void stream_info(void) {
  uint8_t payload[12];
  put32(payload,1000U); put16(payload+4,1U); put16(payload+6,16U); put32(payload+8,dropped);
  enqueue_frame(2U,samples_completed,payload,12U,0U);
}
static void stream_block(uint16_t *data) {
  uint32_t first=samples_completed;
  samples_completed+=HALF_SAMPLES;
  enqueue_frame(1U,first,(const uint8_t*)data,HALF_SAMPLES*2U,HALF_SAMPLES);
}
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc) {
  if (hadc->Instance==ADC1) stream_block(&adc_buffer[0]);
}
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
  if (hadc->Instance==ADC1) stream_block(&adc_buffer[HALF_SAMPLES]);
}
void HAL_ADC_ErrorCallback(ADC_HandleTypeDef *hadc) {
  if (hadc->Instance==ADC1) stream_faults|=2U;
}
void ADC_IRQHandler(void) { HAL_ADC_IRQHandler(&hadc1); }
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance==USART1) { tx_tail++; __DMB(); tx_busy=0U; }
}
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance==USART1) stream_faults|=4U;
}
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance!=USART1) return;
  if (rx_byte=='\n') {
    command_line[command_len]='\0';
    if (!command_overflow && command_len==1U && command_line[0]=='I') stream_info();
    if (!command_overflow && command_len>=3U && command_line[0]=='M' && command_line[1]==',') {
      uint32_t token=0U; uint8_t valid=1U;
      for (uint8_t i=2U;i<command_len;i++) {
        char c=command_line[i];
        if (c<'0'||c>'9'||token>429496729U||(token==429496729U&&c>'5')) { valid=0U; break; }
        token=token*10U+(uint32_t)(c-'0');
      }
      if (valid) {
        uint8_t payload[8]; put32(payload,token); put32(payload+4,HAL_GetTick());
        /* Index of next sample at most recently completed 20-sample boundary.
           This is intentionally a block-quantized marker, not exact ADC timing. */
        enqueue_frame(3U,samples_completed,payload,8U,0U);
      }
    }
    command_len=0U; command_overflow=0U;
  } else if (rx_byte!='\r') {
    if (command_len<sizeof(command_line)-1U) command_line[command_len++]=(char)rx_byte;
    else command_overflow=1U;
  }
  if (HAL_UART_Receive_IT(&huart1,&rx_byte,1U)!=HAL_OK) stream_faults|=4U;
}
static void stream_start(void) {
  HAL_NVIC_SetPriority(ADC_IRQn,0U,0U);
  HAL_NVIC_EnableIRQ(ADC_IRQn);
  if (HAL_ADCEx_Calibration_Start(&hadc1,ADC_CALIB_OFFSET,ADC_SINGLE_ENDED)!=HAL_OK) Error_Handler();
  if (HAL_UART_Receive_IT(&huart1,&rx_byte,1U)!=HAL_OK) Error_Handler();
  stream_info();
  if (HAL_ADC_Start_DMA(&hadc1,(uint32_t*)adc_buffer,ADC_BUF_LEN)!=HAL_OK) Error_Handler();
  if (HAL_TIM_Base_Start(&htim3)!=HAL_OK) Error_Handler();
}
static void stream_poll(void) {
  uint32_t saved=__get_PRIMASK(); __disable_irq();
  if (!tx_busy && tx_tail!=tx_head) {
    uint32_t slot=tx_tail%TX_SLOTS;
    tx_busy=1U;
    if (HAL_UART_Transmit_DMA(&huart1,tx_data[slot],tx_len[slot])!=HAL_OK) {
      tx_busy=0U; stream_faults|=4U;
    }
  }
  __set_PRIMASK(saved);
}
