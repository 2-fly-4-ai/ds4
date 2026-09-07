#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

int main(void) {
    char *options[] = {"ds4-agent", "--model", "qwen.gguf", "--ple", "ple.gguf",
                       "--vision", "mmproj.gguf", "--non-interactive", "-p", "test"};
    agent_config cfg = parse_options((int)(sizeof(options) / sizeof(options[0])), options);
    AGENT_TEST_ASSERT(cfg.engine.ple_path && !strcmp(cfg.engine.ple_path, "ple.gguf"));
    AGENT_TEST_ASSERT(cfg.engine.vision_path && !strcmp(cfg.engine.vision_path, "mmproj.gguf"));
    AGENT_TEST_ASSERT(cfg.engine.model_path && !strcmp(cfg.engine.model_path, "qwen.gguf"));
    char *first = agent_default_cache_dir(&cfg);
    char *same = agent_default_cache_dir(&cfg);
    AGENT_TEST_ASSERT(!strcmp(first, same));
    AGENT_TEST_ASSERT(strstr(first, "/.ds4/kvcache/qwen-ple-") != NULL);
    cfg.engine.model_path = "other-qwen.gguf";
    char *other_model = agent_default_cache_dir(&cfg);
    AGENT_TEST_ASSERT(strcmp(first, other_model) != 0);
    cfg.engine.model_path = "qwen.gguf";
    cfg.engine.ple_path = "other-ple.gguf";
    char *other_ple = agent_default_cache_dir(&cfg);
    AGENT_TEST_ASSERT(strcmp(first, other_ple) != 0);
    cfg.engine.ple_path = NULL;
    char *legacy = agent_default_cache_dir(&cfg);
    AGENT_TEST_ASSERT(strstr(legacy, "/.ds4/kvcache") != NULL);
    AGENT_TEST_ASSERT(strstr(legacy, "qwen-ple-") == NULL);
    free(first); free(same); free(other_model); free(other_ple); free(legacy);
    char temp_model[] = "/tmp/ds4-ple-cache-test-XXXXXX";
    int fd = mkstemp(temp_model);
    AGENT_TEST_ASSERT(fd >= 0);
    if (fd >= 0) {
        cfg.engine.model_path = temp_model;
        cfg.engine.ple_path = "ple.gguf";
        char *before = agent_default_cache_dir(&cfg);
        AGENT_TEST_ASSERT(write(fd, "test", 4) == 4);
        char *after = agent_default_cache_dir(&cfg);
        AGENT_TEST_ASSERT(strcmp(before, after) != 0);
        free(before); free(after);
        close(fd);
        unlink(temp_model);
    }
    ds4_agent_unit_tests_run();
    if (agent_test_failures) {
        fprintf(stderr, "ds4-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("ds4-agent tests: ok");
    return 0;
}
