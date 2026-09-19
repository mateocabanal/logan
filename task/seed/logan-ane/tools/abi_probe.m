#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dump_methods(Class cls, int class_methods) {
    Class target = class_methods ? object_getClass(cls) : cls;
    unsigned int n = 0;
    Method *methods = class_copyMethodList(target, &n);
    for (unsigned int i = 0; i < n; i++) {
        SEL s = method_getName(methods[i]);
        const char *name = sel_getName(s);
        const char *types = method_getTypeEncoding(methods[i]);
        printf("  %c %s  %s\n", class_methods ? '+' : '-', name ?: "?", types ?: "?");
    }
    free(methods);
}

int main(void) {
    const char *frameworks[] = {
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
        "/System/Library/Frameworks/CoreML.framework/CoreML",
    };
    for (unsigned i = 0; i < sizeof(frameworks)/sizeof(frameworks[0]); i++) {
        void *h = dlopen(frameworks[i], RTLD_NOW | RTLD_LOCAL);
        printf("dlopen %s -> %s\n", frameworks[i], h ? "ok" : dlerror());
    }

    int count = objc_getClassList(NULL, 0);
    Class *classes = calloc((size_t)count, sizeof(Class));
    count = objc_getClassList(classes, count);
    for (int i = 0; i < count; i++) {
        const char *name = class_getName(classes[i]);
        if (!name) continue;
        if (strncmp(name, "_ANE", 4) == 0 || strncmp(name, "ANE", 3) == 0) {
            printf("CLASS %s\n", name);
            dump_methods(classes[i], 1);
            dump_methods(classes[i], 0);
        }
    }
    free(classes);
    return 0;
}
