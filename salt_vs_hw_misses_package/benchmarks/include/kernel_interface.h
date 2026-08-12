#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <stddef.h>

// Standard kernel interface
typedef struct {
    const char *name;
    const char *description;
    void (*init)(void);           // Initialize data structures
    void (*execute)(void);        // Run the kernel (this is what gets measured)
    void (*cleanup)(void);        // Clean up resources
    void (*verify)(void);         // Optional: verify correctness
} kernel_t;

#endif // KERNEL_INTERFACE_H
