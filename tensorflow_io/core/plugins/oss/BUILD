licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "oss",
    srcs = [
        "oss_filesystem.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/plugins:plugins_header",
        "@com_google_absl//absl/strings",
        "@aliyun_oss_c_sdk",
    ],
    alwayslink = 1,
)
