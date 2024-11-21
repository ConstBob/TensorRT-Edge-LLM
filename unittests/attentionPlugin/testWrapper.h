#pragma once

#include <functional>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

struct TestCase
{
    std::string testName;
    std::function<void()> function;

    TestCase(std::string const& name, std::function<void()> testFunc)
        : testName(name)
        , function(testFunc)
    {
    }
};

class TestCaseRegistrar
{
public:
    // Register a test case with a fully qualified name: testclass.testname
    static void RegisterTest(std::string const& testName, std::string const& caseName, std::function<void()> func)
    {
        std::string const fullName = testName + "." + caseName;
        GetTestCases().emplace_back(fullName, func);
    }

    // Run tests that match the regular expression applied to the full name (testclass.testname)
    static void RunTests(std::string const& pattern = ".*")
    {
        std::regex regex_pattern(pattern);
        int passed = 0;
        int failed = 0;

        for (auto const& test : GetTestCases())
        {
            if (std::regex_match(test.testName, regex_pattern))
            {
                std::cout << "Running test: " << test.testName << std::endl;
                try
                {
                    test.function();
                    std::cout << "[PASSED] " << test.testName << std::endl;
                    ++passed;
                }
                catch (std::exception const& e)
                {
                    std::cout << "[FAILED] " << test.testName << " - " << e.what() << std::endl;
                    ++failed;
                }
                catch (...)
                {
                    std::cout << "[FAILED] " << test.testName << " Unknown Error" << std::endl;
                    ++failed;
                }
            }
        }

        // Summary
        std::cout << "\nTests completed. " << passed << " passed, " << failed << " failed." << std::endl;
    }

private:
    // Store the list of registered test cases
    static std::vector<TestCase>& GetTestCases()
    {
        static std::vector<TestCase> test_cases;
        return test_cases;
    }
};

#define REGISTER_TEST(testName, caseName, testFunc)                                                                    \
    static bool test_##testName##_##caseName##_registered = []() {                                                     \
        TestCaseRegistrar::RegisterTest(#testName, #caseName, testFunc);                                               \
        return true;                                                                                                   \
    }();

#define TEST_CASE(testName, caseName)                                                                                  \
    void testName##_##caseName();                                                                                      \
    REGISTER_TEST(testName, caseName, testName##_##caseName);                                                          \
    void testName##_##caseName()