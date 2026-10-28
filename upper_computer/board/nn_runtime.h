/* Fixed-shape QDQ executor for the five-block sweep. No dynamic allocation.
 * INT8 coefficients are expanded once for M7 dual 16-bit MAC instructions.
 * Model graph, quantizers and pooling boundaries are exported, not recalibrated.
 */
#include <math.h>
#include <stdint.h>
typedef struct {
  int ic,oc,ih,iw,oh,ow,kh,kw,pad,pool,iz,cz,offset;
  const int8_t *w; const int32_t *b; const float *m; const int8_t *lut;
} Layer;
#include "model_data.h"
static int16_t nn_weights[MODEL_WEIGHTS] __attribute__((section(".AXI"),aligned(32)));
static int8_t nn_a[MODEL_FEATURES] __attribute__((aligned(4)));
static int8_t nn_b[MODEL_FEATURES] __attribute__((aligned(4)));
static int16_t nn_col[MODEL_COLUMN] __attribute__((aligned(4)));
static int8_t nn_input[5600];
static float nn_logits[11], nn_prob[11];

/* ONNX QuantizeLinear uses round-to-nearest, ties-to-even. */
static int round_even(float x) {
  int n=(int)x;
  float d=x-(float)n;
  if(d>0.5f || (d==0.5f && (n&1))) ++n;
  else if(d< -0.5f || (d== -0.5f && (n&1))) --n;
  return n;
}
static int8_t q8(float x, int zero) {
  if (x>255.f) return 127;
  if (x< -255.f) return -128;
  int n=round_even(x)+zero;
  return (int8_t)(n>127?127:(n< -128?-128:n));
}
static void nn_init(void) {
  for(int l=0;l<5;l++) {
    const Layer *c=&model_layers[l];
    int n=c->ic*c->oc*c->kh*c->kw;
    for(int i=0;i<n;i++) nn_weights[c->offset+i]=c->w[i];
  }
}
static void nn_conv(const int8_t *input,int8_t *out,const Layer *c) {
  int n=c->ic*c->kh*c->kw;
  for(int y=0;y<c->oh;y++) for(int x=0;x<c->ow;x++) {
    int8_t best[128];
    for(int o=0;o<c->oc;o++) best[o]=-128;
    for(int p=0;p<c->pool;p++) {
      int col=0, start=x*c->pool+p-c->pad;
      for(int i=0;i<c->ic;i++) for(int ky=0;ky<c->kh;ky++) {
        const int8_t *row=input+(i*c->ih+y+ky)*c->iw;
        for(int k=0;k<c->kw;k++) {
          int xx=start+k;
          nn_col[col++]=(xx<0 || xx>=c->iw)?0:(int)row[xx]-c->iz;
        }
      }
      for(int o=0;o<c->oc;o++) {
        const uint32_t *w=(const uint32_t *)(nn_weights+c->offset+o*n);
        const uint32_t *a=(const uint32_t *)nn_col;
        int32_t sum=c->b[o];
        for(int k=0;k<n/2;k++) sum=(int32_t)__SMLAD(a[k],w[k],(uint32_t)sum);
        int8_t v=q8((float)sum*c->m[o],c->cz);
        if(v>best[o]) best[o]=v;
      }
    }
    for(int o=0;o<c->oc;o++) out[(o*c->oh+y)*c->ow+x]=c->lut[(int)best[o]+128];
  }
}
static void nn_run(void) {
  const int8_t *in=nn_input;
  for(int l=0;l<5;l++) {
    int8_t *out=(l&1)?nn_b:nn_a;
    nn_conv(in,out,&model_layers[l]); in=out;
  }
  int16_t features[MODEL_FC_INPUT];
  for(int c=0;c<MODEL_TAIL_CHANNELS;c++) for(int b=0;b<MODEL_POOL_BINS;b++) {
    int sum=0;
    for(int x=0;x<MODEL_AVG_WIDTH;x++) sum+=(int)in[c*MODEL_TAIL_WIDTH+b*MODEL_AVG_WIDTH+x]-MODEL_TAIL_ZERO;
    features[c*MODEL_POOL_BINS+b]=(int)q8(sum*MODEL_AVG_MULT,MODEL_AVG_ZERO)-MODEL_AVG_ZERO;
  }
  float maximum=-1.e30f, total=0.f;
  for(int o=0;o<11;o++) {
    int32_t sum=fc_b[o];
    for(int i=0;i<MODEL_FC_INPUT;i++) sum+=features[i]*fc_w[o*MODEL_FC_INPUT+i];
    nn_logits[o]=((int)q8(sum*fc_m[o],MODEL_LOGIT_ZERO)-MODEL_LOGIT_ZERO)*MODEL_LOGIT_SCALE;
    if(nn_logits[o]>maximum) maximum=nn_logits[o];
  }
  for(int o=0;o<11;o++) {nn_prob[o]=expf(nn_logits[o]-maximum);total+=nn_prob[o];}
  for(int o=0;o<11;o++) nn_prob[o]/=total;
}
