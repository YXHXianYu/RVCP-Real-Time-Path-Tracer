#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// ===== Configurations =====

const uint32_t WINDOW_WIDTH  = 800;
const uint32_t WINDOW_HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

#ifdef NDEBUG
const bool g_enableValidationLayers = false;
#else
const bool g_enableValidationLayers = true;
#endif

// ===== Utils =====
template <typename... Args> void logError(const char* format, Args... args) {
    fmt::print(stderr, fg(fmt::color::red), format, args...);
}

template <typename... Args> void logInfo(const char* format, Args... args) {
    fmt::print(stdout, fg(fmt::color::green), format, args...);
}

// ===== Proxy function =====
VkResult CreateDebugUtilsMessengerEXT(VkInstance                                instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks*              pAllocator,
                                      VkDebugUtilsMessengerEXT*                 pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) { func(instance, debugMessenger, pAllocator); }
}

// Main
class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    // 1 level

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        if (g_enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
        }

        vkDestroyInstance(m_instance, nullptr);

        glfwDestroyWindow(m_window);

        glfwTerminate();
    }

    // 2 level
    void createInstance(bool verbose = false) {
        if (g_enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{
            .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName   = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = VK_API_VERSION_1_0,
        };

        auto requiredExtensions = getRequiredExtensions();

        if (verbose) {
            uint32_t extensionCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
            std::vector<VkExtensionProperties> extensions(extensionCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

            logInfo("available extensions:\n");
            for (const auto& extension : extensions) {
                logInfo("\t{}\n", extension.extensionName);
            }

            logInfo("required extensions:\n");
            for (const auto& extension : requiredExtensions) {
                logInfo("\t{}\n", extension);
            }
        }

        auto debugCreateInfo = getDebugUtilsMessengerCreateInfo();

        VkInstanceCreateInfo createInfo{
            .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext            = (g_enableValidationLayers ? &debugCreateInfo : nullptr),
            .pApplicationInfo = &appInfo,

            .enabledLayerCount =
                (g_enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0),
            .ppEnabledLayerNames = (g_enableValidationLayers ? validationLayers.data() : nullptr),

            .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data(),

        };

        if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create instance!");
        }
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (std::strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) { return false; }
        }

        return true;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t     glfwExtensionCount = 0;
        const char** glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> requiredExtensions(glfwExtensions,
                                                    glfwExtensions + glfwExtensionCount);

        if (g_enableValidationLayers) {
            requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return requiredExtensions;
    }

    VkDebugUtilsMessengerCreateInfoEXT getDebugUtilsMessengerCreateInfo() {
        return VkDebugUtilsMessengerCreateInfoEXT{
            .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            .pUserData       = nullptr,
        };
    }

    void setupDebugMessenger() {
        if (!g_enableValidationLayers) return;

        auto createInfo = getDebugUtilsMessengerCreateInfo();

        if (CreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_debugMessenger) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug messenger!");
        }
    }

    // static member function
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT             messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            logError("Validation layer: {}\n", pCallbackData->pMessage);
        } else {
            logInfo("Validation layer: {}\n", pCallbackData->pMessage);
        }

        return VK_FALSE;
    }

    // members
    GLFWwindow* m_window;

    VkInstance m_instance;

    VkDebugUtilsMessengerEXT m_debugMessenger;
};

int main() {
    HelloTriangleApplication app;

    std::setlocale(LC_ALL, "en_US.UTF-8");

    try {
        app.run();
    } catch (const std::exception& e) {
        logError("Error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}