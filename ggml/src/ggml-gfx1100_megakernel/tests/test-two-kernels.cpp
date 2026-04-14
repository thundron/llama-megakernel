// test-two-kernels: assert that exactly two megakernels exist
//
// If this test fails, someone added extra kernels. Delete them.
// The architecture is: ONE eval megakernel, ONE prompt megakernel. Nothing else.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // Count .hip files in the megakernel directory (excluding tests/)
    const char * dir = __FILE__;
    fs::path test_path(dir);
    fs::path mk_dir = test_path.parent_path().parent_path(); // up from tests/

    int hip_count = 0;
    const char * expected[] = {"decode.hip", "prefill.hip"};
    bool found_eval = false;
    bool found_prompt = false;

    for (const auto & entry : fs::directory_iterator(mk_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() > 4 && name.substr(name.size() - 4) == ".hip") {
            hip_count++;
            if (name == "decode.hip") found_eval = true;
            if (name == "prefill.hip") found_prompt = true;

            if (name != "decode.hip" && name != "prefill.hip") {
                fprintf(stderr, "FAIL: unexpected kernel file: %s\n", name.c_str());
                fprintf(stderr, "Only decode.hip and prefill.hip are allowed.\n");
                return 1;
            }
        }
    }

    if (hip_count != 2) {
        fprintf(stderr, "FAIL: expected exactly 2 .hip files, found %d\n", hip_count);
        return 1;
    }
    if (!found_eval) {
        fprintf(stderr, "FAIL: decode.hip not found\n");
        return 1;
    }
    if (!found_prompt) {
        fprintf(stderr, "FAIL: prefill.hip not found\n");
        return 1;
    }

    printf("PASS: exactly 2 megakernels (decode.hip, prefill.hip)\n");
    return 0;
}
