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
static int (*create_h3)(const h3_config *, h3_session **);
static const char *(*h3_error)(void);
static void (*destroy_h3)(h3_session *);
static int (*create_krea)(const char *, const char *, const char *, int, const char *, krea2_pipeline **, char *, size_t);
static pthread_barrier_t barrier;
static void *h3_thread(void *unused) {
    (void)unused;
    pthread_barrier_wait(&barrier);
    for (int i = 0; i < 5; i++) {
        h3_config config = {0};
        // Null compiler/cache/source fields select the packaged defaults.
        h3_session *session = (void *)(uintptr_t)1; char error[512] = "old error";
        int status = create_h3(&config, &session);
        snprintf(error, sizeof error, "%s", h3_error());
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
        krea2_pipeline *pipeline = (void *)(uintptr_t)1;
        char error[512] = {0};
        // Explicit absent checkpoints exercise runtime startup and failed model
        // construction without downloading weights or adding a test-only C shim.
        int status = create_krea("/hrx-test-missing.safetensors", "/hrx-test-te.safetensors",
                                 "/hrx-test-vae.safetensors", 1, NULL, &pipeline, error, sizeof error);
        assert(status != 0 && pipeline == NULL && error[0]);

    }
    return NULL;
}
static void *load(const char *path) { void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL); if (!h) fprintf(stderr, "%s\n", dlerror()); assert(h); return h; }
int main(int argc, char **argv) {
    assert(argc == 3);
    void *h3 = load(argv[1]), *krea = load(argv[2]);
    unsigned (*hv)(void) = dlsym(h3, "h3_abi_version");
    unsigned (*kv)(void) = dlsym(krea, "krea2_abi_version");
    assert(hv && kv && hv() == H3_ABI_VERSION && kv() == KREA2_ABI_VERSION);
    create_h3 = dlsym(h3, "h3_create"); destroy_h3 = dlsym(h3, "h3_destroy");
    h3_error = dlsym(h3, "h3_last_error");
    create_krea = dlsym(krea, "krea2_pipeline_create_files");
    assert(create_h3 && destroy_h3 && h3_error && create_krea);
    h3_session *session = NULL;
    assert(create_h3(NULL, &session) == 64 && h3_error()[0]);
    pthread_t a,b; pthread_barrier_init(&barrier, NULL, 2);
    assert(pthread_create(&a,NULL,h3_thread,NULL) == 0);
    assert(pthread_create(&b,NULL,krea_thread,NULL) == 0);
    pthread_join(a,NULL); pthread_join(b,NULL); pthread_barrier_destroy(&barrier);
    dlclose(h3); dlclose(krea);
    puts("C headers, caller errors, concurrent H3/Krea runtimes, drop/reopen: ok");
    return 0;
}
