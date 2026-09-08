// Compile against both generated headers; load independently built model DSOs.
#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "h3.h"
#include "krea2.h"
#include "krea2_pipeline.h"
static int (*create_h3)(const h3_config *, h3_session **, char *, size_t);
static void (*destroy_h3)(h3_session *);
static void *(*allocate)(size_t);
static void (*release)(void *);
static void (*copy)(void *, const void *, size_t);
static pthread_barrier_t barrier;
static void *h3_thread(void *unused) {
    (void)unused;
    pthread_barrier_wait(&barrier);
    for (int i = 0; i < 5; i++) {
        h3_config config = {0};
        // Null compiler/cache/source fields select the packaged defaults.
        h3_session *session = (void *)(uintptr_t)1; char error[512] = "old error";
        int status = create_h3(&config, &session, error, sizeof error);
        if (status) fprintf(stderr, "H3: %s\n", error);
        assert(status == 0 && session && error[0] == 0);
        destroy_h3(session);
    }
    return NULL;
}
static void *krea_thread(void *unused) {
    (void)unused;
    pthread_barrier_wait(&barrier);
    for (int i = 0; i < 5; i++) {
        unsigned char input[4096], output[4096]; memset(input, i + 1, sizeof input);
        void *gpu = allocate(sizeof input); assert(gpu);
        copy(gpu, input, sizeof input); copy(output, gpu, sizeof output);
        assert(memcmp(input, output, sizeof input) == 0); release(gpu);
    }
    return NULL;
}
static void *load(const char *path) { void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL); if (!h) fprintf(stderr, "%s\n", dlerror()); assert(h); return h; }
int main(int argc, char **argv) {
    assert(argc == 4);
    void *h3 = load(argv[1]), *krea = load(argv[2]), *testkit = load(argv[3]);
    unsigned (*hv)(void) = dlsym(h3, "h3_abi_version");
    unsigned (*kv)(void) = dlsym(krea, "krea2_abi_version");
    assert(hv && kv && hv() == H3_ABI_VERSION && kv() == KREA2_ABI_VERSION);
    create_h3 = dlsym(h3, "h3_create_ex"); destroy_h3 = dlsym(h3, "h3_destroy");
    allocate = dlsym(testkit, "test_alloc"); release = dlsym(testkit, "test_free"); copy = dlsym(testkit, "test_copy");
    assert(create_h3 && destroy_h3 && allocate && release && copy);
    char error[128] = {0}; h3_session *session = NULL;
    assert(create_h3(NULL, &session, error, sizeof error) == 64 && error[0]);
    pthread_t a,b; pthread_barrier_init(&barrier, NULL, 2);
    assert(pthread_create(&a,NULL,h3_thread,NULL) == 0);
    assert(pthread_create(&b,NULL,krea_thread,NULL) == 0);
    pthread_join(a,NULL); pthread_join(b,NULL); pthread_barrier_destroy(&barrier);
    dlclose(h3); dlclose(krea); dlclose(testkit);
    puts("C headers, caller errors, concurrent H3/Krea runtimes, drop/reopen: ok");
    return 0;
}
