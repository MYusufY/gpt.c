/* Inference for GPT model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define DEBUG 0

// transformer model

#pragma pack(push, 1)
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int vocab_size;
    int seq_len;
    float norm_eps;
    int use_rope;
    unsigned char shared_weights;
} Config;
#pragma pack(pop)

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    float* positional_embedding_table; // (seq_len, dim)
    float* ln_1_weight; // (layer, dim) layernorm weights
    float* ln_1_bias;   // (layer, dim) layernorm biases
    float* ln_2_weight; // (layer, dim)
    float* ln_2_bias;   // (layer, dim)
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* ln_f_weight; // (dim,)
    float* ln_f_bias;   // (dim,)
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = p->dim;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(p->dim, sizeof(float));
    s->v = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    printf("Memory mapping weights...\n");
    printf("vocab_size: %d, dim: %d\n", p->vocab_size, p->dim);
    printf("seq_len: %d, n_layers: %d\n", p->seq_len, p->n_layers);
    printf("hidden_dim: %d\n", p->hidden_dim);
    
    // 1. token embeddings
    w->token_embedding_table = ptr;
    printf("token_embedding_table at: %p\n", (void*)w->token_embedding_table);
    ptr += p->vocab_size * p->dim;
    
    // 2. positional embeddings
    w->positional_embedding_table = ptr;
    printf("positional_embedding_table at: %p\n", (void*)w->positional_embedding_table);
    ptr += p->seq_len * p->dim;
    
    // 3. allocate space for layer weights
    w->ln_1_weight = malloc(p->n_layers * p->dim * sizeof(float));
    w->ln_1_bias = malloc(p->n_layers * p->dim * sizeof(float));
    w->wq = malloc(p->n_layers * p->dim * p->dim * sizeof(float));
    w->wk = malloc(p->n_layers * p->dim * p->dim * sizeof(float));
    w->wv = malloc(p->n_layers * p->dim * p->dim * sizeof(float));
    w->wo = malloc(p->n_layers * p->dim * p->dim * sizeof(float));
    w->ln_2_weight = malloc(p->n_layers * p->dim * sizeof(float));
    w->ln_2_bias = malloc(p->n_layers * p->dim * sizeof(float));
    w->w1 = malloc(p->n_layers * p->dim * p->hidden_dim * sizeof(float));
    w->w2 = malloc(p->n_layers * p->hidden_dim * p->dim * sizeof(float));
    
    // 4. copy weights in the same order as version3_export (for GPT)
    for (int l = 0; l < p->n_layers; l++) {
        // LayerNorm 1 (attention norm)
        memcpy(w->ln_1_weight + l * p->dim, ptr, p->dim * sizeof(float));
        ptr += p->dim;
        
        memcpy(w->ln_1_bias + l * p->dim, ptr, p->dim * sizeof(float));
        ptr += p->dim;
        
        // attention weights
        memcpy(w->wq + l * p->dim * p->dim, ptr, p->dim * p->dim * sizeof(float));
        ptr += p->dim * p->dim;
        
        memcpy(w->wk + l * p->dim * p->dim, ptr, p->dim * p->dim * sizeof(float));
        ptr += p->dim * p->dim;
        
        memcpy(w->wv + l * p->dim * p->dim, ptr, p->dim * p->dim * sizeof(float));
        ptr += p->dim * p->dim;
        
        memcpy(w->wo + l * p->dim * p->dim, ptr, p->dim * p->dim * sizeof(float));
        ptr += p->dim * p->dim;
        
        // LayerNorm 2 (FFN norm)
        memcpy(w->ln_2_weight + l * p->dim, ptr, p->dim * sizeof(float));
        ptr += p->dim;
        
        memcpy(w->ln_2_bias + l * p->dim, ptr, p->dim * sizeof(float));
        ptr += p->dim;
        
        // FFN weights
        memcpy(w->w1 + l * p->dim * p->hidden_dim, ptr, p->dim * p->hidden_dim * sizeof(float));
        ptr += p->dim * p->hidden_dim;
        
        memcpy(w->w2 + l * p->hidden_dim * p->dim, ptr, p->hidden_dim * p->dim * sizeof(float));
        ptr += p->hidden_dim * p->dim;
    }
    
    // 5. final LayerNorm
    w->ln_f_weight = ptr;
    ptr += p->dim;
    w->ln_f_bias = ptr;
    ptr += p->dim;
    
    // 6. classifier weights
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
    
    printf("Weight mapping complete!\n");
}

void free_weights(TransformerWeights* w) {
    if (w->ln_1_weight) free(w->ln_1_weight);
    if (w->ln_1_bias) free(w->ln_1_bias);
    if (w->wq) free(w->wq);
    if (w->wk) free(w->wk);
    if (w->wv) free(w->wv);
    if (w->wo) free(w->wo);
    if (w->ln_2_weight) free(w->ln_2_weight);
    if (w->ln_2_bias) free(w->ln_2_bias);
    if (w->w1) free(w->w1);
    if (w->w2) free(w->w2);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { printf("Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    
    unsigned int magic;
    if (fread(&magic, sizeof(unsigned int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic != 0x616b3432) {
        printf("Invalid magic number: 0x%08X (expected 0x616b3432)\n", magic);
        exit(EXIT_FAILURE);
    }
    
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 3) {
        printf("Unsupported version: %d (expected 3)\n", version);
        exit(EXIT_FAILURE);
    }
    
    if (fread(&config->dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->hidden_dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->n_layers, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->n_heads, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->vocab_size, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->seq_len, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->norm_eps, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->use_rope, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    
    unsigned char shared_weights_flag;
    if (fread(&shared_weights_flag, sizeof(unsigned char), 1, file) != 1) exit(EXIT_FAILURE);
    int shared_weights = (int)shared_weights_flag;
    
    long current_pos = ftell(file);
    if (current_pos < 256) {
        fseek(file, 256 - current_pos, SEEK_CUR);
    }
    
    printf("Raw config values: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, vocab_size=%d, seq_len=%d, norm_eps=%f, use_rope=%d\n",
           config->dim, config->hidden_dim, config->n_layers, config->n_heads,
           config->vocab_size, config->seq_len, config->norm_eps, config->use_rope);
    
    printf("After processing: vocab_size=%d, shared_weights=%d\n", config->vocab_size, shared_weights);
    
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { printf("open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { printf("mmap failed!\n"); exit(EXIT_FAILURE); }
    
    float* weights_ptr = *data + 256/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    printf("Building transformer from: %s\n", checkpoint_path);
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    printf("Checkpoint loaded successfully, file size: %zd bytes\n", t->file_size);
    printf("Data mapped at: %p\n", (void*)t->data);
    
    printf("Allocating run state...\n");
    malloc_run_state(&t->state, &t->config);
    printf("Run state allocated successfully!\n");

    printf("Testing weight access...\n");
    float test_value = t->weights.token_embedding_table[0];
    printf("First weight value: %f\n", test_value);
}

void free_transformer(Transformer* t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_weights(&t->weights);
    free_run_state(&t->state);
}

// neural net blocks; the dynamics of the Transformer

void layernorm(float* o, float* x, float* weight, float* bias, int size) {
    // calculate mean
    float mean = 0.0f;
    for (int j = 0; j < size; j++) {
        mean += x[j];
    }
    mean /= size;
    
    // calculate variance
    float variance = 0.0f;
    for (int j = 0; j < size; j++) {
        variance += (x[j] - mean) * (x[j] - mean);
    }
    variance /= size;
    variance += 1e-5f;
    
    // normalize and scale
    float inv_std = 1.0f / sqrtf(variance);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * ((x[j] - mean) * inv_std) + bias[j];
    }
}

void softmax(float* x, int size) {
    // find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    #if DEBUG
    // debug output section
    printf("=== FORWARD PASS START ===\n");
    printf("token=%d, pos=%d\n", token, pos);
    
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    printf("dim=%d, hidden_dim=%d, head_size=%d\n", dim, hidden_dim, head_size);
    
    // 1. token embedding lookup
    printf("1. Token embedding lookup...\n");
    printf("token_embedding_table: %p, token: %d, dim: %d\n", 
           (void*)w->token_embedding_table, token, dim);
    
    // validate token is within vocabulary
    if (token < 0 || token >= p->vocab_size) {
        printf("ERROR: Token %d out of vocabulary range [0, %d]\n", token, p->vocab_size - 1);
        exit(EXIT_FAILURE);
    }
    
    float* content_row = w->token_embedding_table + token * dim;
    printf("content_row: %p\n", (void*)content_row);
    memcpy(x, content_row, dim * sizeof(float));
    printf("Token embedding done.\n");
    
    // 2. add positional embedding (GPT style: learned, not RoPE)
    printf("2. Positional embedding...\n");
    if (pos < p->seq_len) {
        printf("pos %d < seq_len %d\n", pos, p->seq_len);
        float* pos_row = w->positional_embedding_table + pos * dim;
        printf("pos_row: %p\n", (void*)pos_row);
        for (int i = 0; i < dim; i++) {
            x[i] += pos_row[i];
        }
    } else {
        printf("pos %d >= seq_len %d, skipping positional embedding\n", pos, p->seq_len);
    }
    printf("Positional embedding done.\n");
    
    // 3. forward pass through all layers
    printf("3. Processing %d layers...\n", p->n_layers);
    for (int l = 0; l < p->n_layers; l++) {
        printf("  Layer %d...\n", l);
        
        long long layer_offset = l * dim;
        long long layer_offset_attn = l * dim * dim;
        long long layer_offset_ffn = l * dim * hidden_dim;
        
        printf("    Offsets: norm=%lld, attn=%lld, ffn=%lld\n", 
               layer_offset, layer_offset_attn, layer_offset_ffn);

        printf("    3a. LayerNorm...\n");
        layernorm(s->xb, x, 
                 w->ln_1_weight + layer_offset, 
                 w->ln_1_bias + layer_offset, 
                 dim);
        printf("    LayerNorm done.\n");
        
        printf("    3b. QKV projections...\n");
        matmul(s->q, s->xb, w->wq + layer_offset_attn, dim, dim);
        matmul(s->k, s->xb, w->wk + layer_offset_attn, dim, dim); 
        matmul(s->v, s->xb, w->wv + layer_offset_attn, dim, dim);
        printf("    QKV projections done.\n");
        
        printf("    3c. KV cache...\n");
        long long cache_offset = l * p->seq_len * dim + pos * dim;
        printf("    cache_offset: %lld\n", cache_offset);
        memcpy(s->key_cache + cache_offset, s->k, dim * sizeof(float));
        memcpy(s->value_cache + cache_offset, s->v, dim * sizeof(float));
        printf("    KV cache done.\n");
        
        printf("    3d. Attention...\n");
        memset(s->xb, 0, dim * sizeof(float));
        
        for (int h = 0; h < p->n_heads; h++) {
            printf("      Head %d...\n", h);
            float* q = s->q + h * head_size;            
            float* att = s->att + h * p->seq_len;

            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + l * p->seq_len * dim + t * dim + h * head_size;
                
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);            
            float* xb_head = s->xb + h * head_size;
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + l * p->seq_len * dim + t * dim + h * head_size;
                
                float a = att[t];
                
                for (int i = 0; i < head_size; i++) {
                    xb_head[i] += a * v[i];
                }
            }
        }
        printf("    Attention done.\n");
        
        printf("    3e. Output projection...\n");
        matmul(s->xb2, s->xb, w->wo + layer_offset_attn, dim, dim);
        printf("    Output projection done.\n");
        
        printf("    3f. Residual connection...\n");
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        printf("    Residual connection done.\n");
        
        printf("    3g. FFN LayerNorm...\n");
        layernorm(s->xb, x, 
                 w->ln_2_weight + layer_offset, 
                 w->ln_2_bias + layer_offset, 
                 dim);
        printf("    FFN LayerNorm done.\n");
        
        printf("    3h. FFN...\n");

        matmul(s->hb, s->xb, w->w1 + layer_offset_ffn, dim, hidden_dim);
        
        matmul(s->xb, s->hb, w->w2 + layer_offset_ffn, hidden_dim, dim);
        printf("    FFN done.\n");
        
        printf("    3i. FFN Residual...\n");
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
        printf("    FFN Residual done.\n");
        
        printf("  Layer %d complete.\n", l);
    }
    
    // 4. final LayerNorm
    printf("4. Final LayerNorm...\n");
    layernorm(x, x, w->ln_f_weight, w->ln_f_bias, dim);
    printf("Final LayerNorm done.\n");
    
    // 5. classifier (output projection to vocabulary)
    printf("5. Classifier...\n");
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    printf("Classifier done.\n");
    
    printf("=== FORWARD PASS COMPLETE ===\n");
    return s->logits;
    #else
    // no debug output section, prints just the response :) (default)
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    // validate token is within vocabulary
    if (token < 0 || token >= p->vocab_size) {
        printf("ERROR: Token %d out of vocabulary range [0, %d]\n", token, p->vocab_size - 1);
        exit(EXIT_FAILURE);
    }
    
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));
    
    // 2. add positional embedding (GPT style: learned, not RoPE)
    if (pos < p->seq_len) {
        float* pos_row = w->positional_embedding_table + pos * dim;
        for (int i = 0; i < dim; i++) {
            x[i] += pos_row[i];
        }
    }
    
    // 3. forward pass through all layers
    for (int l = 0; l < p->n_layers; l++) {
        long long layer_offset = l * dim;
        long long layer_offset_attn = l * dim * dim;
        long long layer_offset_ffn = l * dim * hidden_dim;
        
        layernorm(s->xb, x, 
                 w->ln_1_weight + layer_offset, 
                 w->ln_1_bias + layer_offset, 
                 dim);
        
        matmul(s->q, s->xb, w->wq + layer_offset_attn, dim, dim);
        matmul(s->k, s->xb, w->wk + layer_offset_attn, dim, dim); 
        matmul(s->v, s->xb, w->wv + layer_offset_attn, dim, dim);
        
        long long cache_offset = l * p->seq_len * dim + pos * dim;
        memcpy(s->key_cache + cache_offset, s->k, dim * sizeof(float));
        memcpy(s->value_cache + cache_offset, s->v, dim * sizeof(float));
        
        memset(s->xb, 0, dim * sizeof(float));
        
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            
            float* att = s->att + h * p->seq_len;
            
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + l * p->seq_len * dim + t * dim + h * head_size;
                
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            
            softmax(att, pos + 1);
            
            float* xb_head = s->xb + h * head_size;
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + l * p->seq_len * dim + t * dim + h * head_size;
                
                float a = att[t];
                
                for (int i = 0; i < head_size; i++) {
                    xb_head[i] += a * v[i];
                }
            }
        }
        
        matmul(s->xb2, s->xb, w->wo + layer_offset_attn, dim, dim);
        
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        
        layernorm(s->xb, x, 
                 w->ln_2_weight + layer_offset, 
                 w->ln_2_bias + layer_offset, 
                 dim);
        

        matmul(s->hb, s->xb, w->w1 + layer_offset_ffn, dim, hidden_dim);
        
        matmul(s->xb, s->hb, w->w2 + layer_offset_ffn, hidden_dim, dim);
        
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    
    // 4. final LayerNorm
    layernorm(x, x, w->ln_f_weight, w->ln_f_bias, dim);
    
    // 5. classifier (output projection to vocabulary)
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);

    return s->logits;
    #endif
}

// the BPE (Byte Pair Encoding) tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

// GELU activation function (for GPT format)
float gelu(float x) {
    // GELU approx.: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}


void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        printf("Token %d: %s\n", i, t->vocab[i]);
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { printf("couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { printf("failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { printf("failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { printf("failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { printf("failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { printf("cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        tokens[best_idx] = best_id;

        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        
        softmax(logits, sampler->vocab_size);
        
        float coin = random_f32(&sampler->rng_state);

        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// utilities: time

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 0, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    
    printf("Encoded prompt tokens: ");
    for (int i = 0; i < num_prompt_tokens; i++) {
        printf("%d ", prompt_tokens[i]);
    }
    printf("\n");

    // start the main loop
    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s
    if (pos > 1) {
        long end = time_in_ms();
        printf("achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// chat loop

// Andrej Karpathy's original comment on chat feature of Llama2.c:
//  I manually inspected the tokens for a few chat conversations compared to
//  python reference and that seemed ok, but this was not thoroughly tested and
//  is not safely implemented, it's more a proof of concept atm.

// My (Yusuf Yıldırım) comment on chat feature of GPT.c:
//  Just like original Llama2-compatible version, its not tested and just... exists :)
//  I didnt edit it, but i decided to keep it. So as Karpathy says:
//      "is not safely implemented, it's more a proof of concept atm."

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are somewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {
        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// CLI, include only if not testing

#ifndef TESTING

void error_usage() {
    printf("Usage:   run <checkpoint> [options]\n");
    printf("Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    printf("Options:\n");
    printf("  -t <float>  temperature in [0,inf], default 1.0\n");
    printf("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    printf("  -s <int>    random seed, default time(NULL)\n");
    printf("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    printf("  -i <string> input prompt\n");
    printf("  -z <string> optional path to custom tokenizer\n");
    printf("  -m <string> mode: generate|chat, default: generate\n");
    printf("  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        printf("unknown mode: %s\n", mode);
        error_usage();
    }

    // cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif