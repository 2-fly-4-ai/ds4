/* Metadata/tokenizer-only test: no weight reads, CPU inference, or GPU. */
#include "../ds4.c"
#include <assert.h>

int main(int argc, char **argv) {
    assert(argc == 2);
    ds4_model model = {0};
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    assert(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN);
    ds4_str pre = {0};
    assert(model_get_string(&model, "tokenizer.ggml.pre", &pre));
    printf("pre=%.*s\n", (int)pre.len, pre.ptr);
    assert(ds4_streq(pre, "qwen35"));
    ds4_engine e = {0};
    vocab_load(&e.vocab, &model);
    assert(e.vocab.bos_id == -1);
    assert(e.vocab.im_start_id == vocab_lookup(&e.vocab, "<|im_start|>"));
    assert(e.vocab.eos_id == vocab_lookup(&e.vocab, "<|im_end|>"));
    assert(ds4_token_is_stop(&e, e.vocab.im_end_id));
    assert(ds4_token_is_stop(&e, e.vocab.endoftext_id));
    token_vec rendered = {0}, direct = {0};
    tokenize_rendered_chat_vocab(&e.vocab,
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", &rendered);
    encode_chat_prompt(&e.vocab, NULL, "Hello", DS4_THINK_NONE, &direct);
    assert(rendered.len == direct.len);
    assert(!memcmp(rendered.v, direct.v, rendered.len * sizeof(int)));
    assert(rendered.v[0] == e.vocab.im_start_id);
    token_vec actual = {0}, reference = {0};
    const char *text = "1234567 function();\n\n café 🐈\n";
    bpe_tokenize_text(&e.vocab, text, &actual);
    bpe_tokenize_text_qwen35(&e.vocab, text, &reference);
    assert(actual.len == reference.len);
    assert(!memcmp(actual.v, reference.v, actual.len * sizeof(int)));
    printf("ChatML and pre-tokenizer dispatch PASS; start=%d end=%d text_end=%d\n",
           e.vocab.im_start_id, e.vocab.im_end_id, e.vocab.endoftext_id);
    ds4_tokens_free(&rendered); ds4_tokens_free(&direct);
    ds4_tokens_free(&actual); ds4_tokens_free(&reference);
    vocab_free(&e.vocab); model_close(&model);
    return 0;
}
