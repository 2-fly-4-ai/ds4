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
    ds4_agent_unit_tests_run();
    if (agent_test_failures) {
        fprintf(stderr, "ds4-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("ds4-agent tests: ok");
    return 0;
}
