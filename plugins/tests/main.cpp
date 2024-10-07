#include "testWrapper.h"

std::string parseTestFilterFromArgs(int argc, char* argv[]) {
    std::string filter = ".*";  // Default pattern (match all tests)
    for (int i = 1; i < argc; ++i) {
        std::string const arg = std::string(argv[i]);
        if (arg.find("--test_filter=") == 0) {
            filter = arg.substr(std::string("--test_filter=").length());
        }
    }
    return filter;
}

int main(int argc, char* argv[]) {
    // Parse the command-line arguments for the test filter
    std::string test_filter = parseTestFilterFromArgs(argc, argv);
    
    // Run the tests based on the filter
    TestCaseRegistrar::RunTests(test_filter);
    
    return 0;
}