/* Included by generated main.c. Sampling/filtering runs in the ADC ISR;
 * inference runs in the foreground. DMA SRAM is MPU normal non-cacheable.
 * The classifier never receives cue labels. */
#include "nn_runtime.h"
#define SAMPLE_RATE_HZ 2000U
#define ADC_CHANNELS 2U
#define ADC_BUF_LEN 80U
#define HALF_SAMPLES 20U
#define TX_SLOTS 32U
#define FRAME_MAX 256U
static uint16_t adc_buffer[ADC_BUF_LEN] __attribute__((section(".RAM_D2"),aligned(32)));
static uint8_t tx_data[TX_SLOTS][FRAME_MAX] __attribute__((section(".RAM_D2"),aligned(32)));
static uint16_t tx_len[TX_SLOTS];
static volatile uint32_t tx_head,tx_tail,frame_seq,samples_completed,dropped;
static volatile uint16_t stream_faults;
static volatile uint8_t tx_busy,run_mode=1,mode_request=255,info_request;
static uint8_t rx_byte,command_len;
static char command_line[32];
static uint8_t command_overflow;
static int8_t signal_ring[2][2800];
static float filter_state[2][5][2];
/* Display branch: trailing 200 ms mean absolute filtered ADC counts.
 * Independent of model normalization and classification. */
static float envelope_ring[2][400];
static double envelope_sum[2];
static uint32_t envelope_pos,envelope_count;
static uint32_t ring_pos,ring_count,filter_cycles,clipped_samples;
static uint8_t filter_started;
static volatile uint32_t ready_end,ready_filter_us;
static uint32_t last_end,skipped,over100,over200,max_total_us;
static uint8_t armed,rest_count,word_count,candidate=255;
static uint32_t last_emit,decoder_end;
static uint8_t emitted_once;
/* Validation-only binary input, V + newline then 5600 int8 bytes + CRC16.
 * It disables live output until N; responses are explicitly kind 6. */
static volatile uint32_t test_remaining,test_pos,test_started,test_length;
static volatile uint8_t test_pending;
static volatile uint8_t test_raw;
static uint8_t test_packet[11202] __attribute__((aligned(4)));
static void stream_poll(void);
static void put16(uint8_t *p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);}
static void put32(uint8_t *p,uint32_t v){put16(p,(uint16_t)v);put16(p+2,(uint16_t)(v>>16));}
static uint16_t crc16(const uint8_t *p,uint16_t n){
  uint16_t crc=0xffff;
  for(uint16_t i=0;i<n;i++) {crc^=(uint16_t)p[i]<<8;for(uint8_t j=0;j<8;j++)crc=(crc&0x8000)?(uint16_t)((crc<<1)^0x1021):(uint16_t)(crc<<1);}
  return crc;
}
static void enqueue_frame(uint8_t type,uint32_t first,const uint8_t *payload,uint16_t len,uint16_t count){
  uint32_t saved=__get_PRIMASK();__disable_irq();
  uint32_t seq=frame_seq++;
  if(len>FRAME_MAX-20 || (uint32_t)(tx_head-tx_tail)>=TX_SLOTS){dropped++;stream_faults|=1;__set_PRIMASK(saved);return;}
  uint32_t slot=tx_head%TX_SLOTS;uint8_t *p=tx_data[slot];
  p[0]=0xa5;p[1]=0x5a;p[2]=2;p[3]=type;put16(p+4,len);put32(p+6,seq);put32(p+10,first);
  put16(p+14,count);put16(p+16,stream_faults);memcpy(p+18,payload,len);
  put16(p+18+len,crc16(p,18+len));tx_len[slot]=20+len;__DMB();tx_head++;__set_PRIMASK(saved);
  stream_poll();
}
static void stream_info(void){
  uint8_t payload[12];put32(payload,2000);put16(payload+4,2);put16(payload+6,16);put32(payload+8,dropped);
  enqueue_frame(2,samples_completed,payload,12,0);
  char text[220];
  int n=snprintf(text,sizeof(text),"{\"model\":\"%s\",\"sha\":\"%s\",\"fs\":2000,\"win\":2800,\"step\":200,\"mode\":%u,\"fw\":1,\"env_hz\":100,\"env_ms\":200}",MODEL_NAME,MODEL_HASH,run_mode);
  if(n>0 && n<(int)sizeof(text))enqueue_frame(5,samples_completed,(const uint8_t*)text,(uint16_t)n,0);
}
static void decoder_reset(void){armed=rest_count=word_count=emitted_once=0;candidate=255;decoder_end=last_emit=0;}
static void pipeline_reset(void){
  ring_pos=ring_count=filter_cycles=clipped_samples=0;filter_started=0;
  envelope_pos=envelope_count=0;envelope_sum[0]=envelope_sum[1]=0.;
  memset(envelope_ring,0,sizeof(envelope_ring));
  ready_end=last_end=0;skipped=over100=over200=max_total_us=0;decoder_reset();
}
static void filter_block(const uint16_t *raw){
  uint32_t started=TIM2->CNT;
  for(int t=0;t<20;t++){
    for(int ch=0;ch<2;ch++){
      float x=(float)raw[t*2+ch]-32768.f;
      if(raw[t*2+ch]<=1 || raw[t*2+ch]>=65534)clipped_samples++;
      if(!filter_started)for(int s=0;s<5;s++)for(int j=0;j<2;j++)filter_state[ch][s][j]=filter_zi[s*2+j]*x;
      for(int s=0;s<5;s++){
        const float *a=filter_sos+s*6;float *z=filter_state[ch][s];
        float y=a[0]*x+z[0];z[0]=a[1]*x-a[4]*y+z[1];z[1]=a[2]*x-a[5]*y;x=y;
      }
      signal_ring[ch][ring_pos]=q8(((x-norm_mean[ch])/norm_scale[ch])/MODEL_INPUT_SCALE,MODEL_INPUT_ZERO);
      float absolute=fabsf(x);
      envelope_sum[ch]+=(double)absolute-(double)envelope_ring[ch][envelope_pos];
      envelope_ring[ch][envelope_pos]=absolute;
    }
    envelope_pos++;if(envelope_pos==400)envelope_pos=0;
    if(envelope_count<400)envelope_count++;
    filter_started=1;ring_pos++;if(ring_pos==2800)ring_pos=0;
    if(ring_count<2800)ring_count++;
  }
  if(run_mode==1){
    float envelope[2]={(float)(envelope_sum[0]/envelope_count),(float)(envelope_sum[1]/envelope_count)};
    enqueue_frame(8,samples_completed,(const uint8_t*)envelope,8,1);
  }
  filter_cycles+=TIM2->CNT-started;
  if(samples_completed%200==0){
    ready_filter_us=filter_cycles;filter_cycles=0;
    if(ring_count==2800){__DMB();ready_end=samples_completed;}
  }
}
static void stream_block(uint16_t *data){
  uint32_t first=samples_completed;samples_completed+=20;
  if(run_mode==0)enqueue_frame(1,first,(const uint8_t*)data,80,20);
  else if(run_mode==1)filter_block(data);
}
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *h){if(h->Instance==ADC1)stream_block(adc_buffer);}
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *h){if(h->Instance==ADC1)stream_block(adc_buffer+40);}
void HAL_ADC_ErrorCallback(ADC_HandleTypeDef *h){if(h->Instance==ADC1)stream_faults|=2;}
void ADC_IRQHandler(void){HAL_ADC_IRQHandler(&hadc1);}
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *h){if(h->Instance==USART1){tx_tail++;__DMB();tx_busy=0;stream_poll();}}
void HAL_UART_ErrorCallback(UART_HandleTypeDef *h){if(h->Instance==USART1)stream_faults|=4;}
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *h){
  if(h->Instance!=USART1)return;
  if(test_remaining){
    test_packet[test_pos++]=rx_byte;
    if(--test_remaining==0){__DMB();test_pending=1;}
  }else if(rx_byte=='\n'){
    command_line[command_len]=0;
    if(!command_overflow){
      if(!strcmp(command_line,"I"))info_request=1;
      else if(!strcmp(command_line,"C"))mode_request=0;
      else if(!strcmp(command_line,"N"))mode_request=1;
      else if(!strcmp(command_line,"R")){mode_request=2;}
      else if((!strcmp(command_line,"V") || !strcmp(command_line,"F")) && !test_pending){
        run_mode=2;test_raw=(command_line[0]=='F');test_length=test_raw?11200:5600;
        test_pos=0;test_remaining=test_length+2;test_started=HAL_GetTick();
      }
      else if(command_len>=3 && command_line[0]=='M' && command_line[1]==','){
        uint32_t token=0;uint8_t valid=1;
        for(int i=2;i<command_len;i++){char c=command_line[i];if(c<'0'||c>'9'||token>429496729U||(token==429496729U&&c>'5')){valid=0;break;}token=token*10+(uint32_t)(c-'0');}
        if(valid){uint8_t p[8];put32(p,token);put32(p+4,HAL_GetTick());enqueue_frame(3,samples_completed,p,8,0);}
      }
    }
    command_len=command_overflow=0;
  }else if(rx_byte!='\r'){
    if(command_len<sizeof(command_line)-1)command_line[command_len++]=(char)rx_byte;else command_overflow=1;
  }
  if(HAL_UART_Receive_IT(&huart1,&rx_byte,1)!=HAL_OK)stream_faults|=4;
}
static void stream_start(void){
  /* Dedicated free-running 1 MHz timer: independent of debugger/DWT lock. */
  __HAL_RCC_TIM2_CLK_ENABLE();
  TIM2->PSC=(HAL_RCC_GetPCLK1Freq()*2U)/1000000U-1U;
  TIM2->ARR=0xffffffffU;TIM2->EGR=TIM_EGR_UG;TIM2->CR1=TIM_CR1_CEN;
  nn_init();pipeline_reset();
  HAL_NVIC_SetPriority(ADC_IRQn,0,0);HAL_NVIC_EnableIRQ(ADC_IRQn);
  if(HAL_ADCEx_Calibration_Start(&hadc1,ADC_CALIB_OFFSET,ADC_SINGLE_ENDED)!=HAL_OK)Error_Handler();
  if(HAL_UART_Receive_IT(&huart1,&rx_byte,1)!=HAL_OK)Error_Handler();
  stream_info();
  if(HAL_ADC_Start_DMA(&hadc1,(uint32_t*)adc_buffer,ADC_BUF_LEN)!=HAL_OK)Error_Handler();
  if(HAL_TIM_Base_Start(&htim3)!=HAL_OK)Error_Handler();
}
static void stream_poll(void){
  uint32_t saved=__get_PRIMASK();__disable_irq();
  if(!tx_busy && tx_tail!=tx_head){uint32_t slot=tx_tail%TX_SLOTS;tx_busy=1;
    if(HAL_UART_Transmit_DMA(&huart1,tx_data[slot],tx_len[slot])!=HAL_OK){tx_busy=0;stream_faults|=4;}}
  __set_PRIMASK(saved);
}
static uint8_t decode_command(uint32_t end,int label){
  if(decoder_end && end-decoder_end!=200){rest_count=word_count=0;candidate=255;}
  decoder_end=end;
  if(label==10 && nn_prob[label]>=.5f){
    rest_count++;candidate=255;word_count=0;
    if(rest_count>=2 && (!emitted_once || end-last_emit>=1000))armed=1;
    if(rest_count>2)rest_count=2;
    return 255;
  }
  rest_count=0;
  if(!armed || label==10 || nn_prob[label]<.5f){word_count=0;candidate=255;return 255;}
  word_count=(candidate==label)?word_count+1:1;candidate=(uint8_t)label;
  if(word_count<2)return 255;
  armed=0;last_emit=end;emitted_once=1;candidate=255;word_count=0;return (uint8_t)label;
}
static void realtime_poll(void){
  if(test_remaining && HAL_GetTick()-test_started>3000){test_remaining=0;test_pending=0;command_len=0;info_request=1;}
  if(mode_request!=255){
    uint32_t saved=__get_PRIMASK();__disable_irq();run_mode=mode_request;mode_request=255;pipeline_reset();__set_PRIMASK(saved);stream_info();
  }
  if(info_request){info_request=0;stream_info();}
  if(test_pending){
    uint16_t crc=(uint16_t)test_packet[test_length]|((uint16_t)test_packet[test_length+1]<<8);
    if(crc==crc16(test_packet,(uint16_t)test_length)){
      if(test_raw){
        for(int i=0;i<2800;i+=20)filter_block((const uint16_t*)test_packet+i*2);
        for(int ch=0;ch<2;ch++){
          memcpy(nn_input+ch*2800,signal_ring[ch]+ring_pos,2800-ring_pos);
          memcpy(nn_input+ch*2800+2800-ring_pos,signal_ring[ch],ring_pos);
        }
      }else memcpy(nn_input,test_packet,5600);
      uint32_t start=TIM2->CNT;nn_run();
      uint8_t p[92];put32(p,TIM2->CNT-start);memcpy(p+4,nn_logits,44);memcpy(p+48,nn_prob,44);
      enqueue_frame(6,samples_completed,p,92,0);
    }else {uint8_t error=1;enqueue_frame(7,samples_completed,&error,1,0);}
    test_pending=0;return;
  }
  if(run_mode!=1 || !ready_end || ready_end==last_end)return;
  uint32_t start=TIM2->CNT;
  uint32_t saved=__get_PRIMASK();__disable_irq();
  /* Capture at the exact latest completed 200-sample grid. If the main loop
   * arrives after intervening ADC blocks, wait for the next grid, never relabel
   * a later window with an earlier timestamp. */
  if(samples_completed!=ready_end){__set_PRIMASK(saved);return;}
  uint32_t end=ready_end,filter_us=ready_filter_us;
  for(int ch=0;ch<2;ch++){
    memcpy(nn_input+ch*2800,signal_ring[ch]+ring_pos,2800-ring_pos);
    memcpy(nn_input+ch*2800+2800-ring_pos,signal_ring[ch],ring_pos);
  }
  __set_PRIMASK(saved);
  if(last_end && end-last_end>200)skipped+=(end-last_end)/200-1;
  last_end=end;
  uint32_t before=TIM2->CNT;nn_run();uint32_t infer_us=TIM2->CNT-before;
  int top=0;for(int i=1;i<11;i++)if(nn_prob[i]>=nn_prob[top])top=i;
  uint8_t emit=decode_command(end,top);
  uint32_t total_us=(TIM2->CNT-start)+filter_us;
  if(total_us>max_total_us)max_total_us=total_us;if(total_us>=100000)over100++;if(total_us>=200000)over200++;
  uint8_t p[92];put32(p,HAL_GetTick());put32(p+4,infer_us);put32(p+8,total_us);put32(p+12,filter_us);
  put32(p+16,skipped);put32(p+20,max_total_us);put32(p+24,over100);put32(p+28,over200);put32(p+32,dropped);
  p[36]=run_mode;p[37]=(uint8_t)top;p[38]=emit;p[39]=armed;memcpy(p+40,nn_prob,44);
  put32(p+84,clipped_samples);put32(p+88,samples_completed-end);
  enqueue_frame(4,end,p,92,0);
}
