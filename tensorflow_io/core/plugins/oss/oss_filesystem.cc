/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "aos_status.h"
#include "oss_api.h"
#include "oss_auth.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_io/core/plugins/file_system_plugins.h"

namespace tensorflow {
namespace io {
namespace oss {

constexpr char kOSSCredentialsDefaultFile[] = ".osscredentials";
constexpr char kOSSCredentialsFileEnvKey[] = "OSS_CREDENTIALS";
constexpr char kOSSCredentialsSection[] = "OSSCredentials";
constexpr char kOSSCredentialsHostKey[] = "host";
constexpr char kOSSCredentialsAccessIdKey[] = "accessid";
constexpr char kOSSCredentialsAccesskeyKey[] = "accesskey";
constexpr char kOSSAccessIdKey[] = "id";
constexpr char kOSSAccessKeyKey[] = "key";
constexpr char kOSSHostKey[] = "host";
constexpr char kDelim[] = "/";

// The number of bytes for each upload part. Defaults to 64MB
constexpr size_t kUploadPartBytes = 64 * 1024 * 1024;
// The number of bytes to read ahead for buffering purposes
// in the RandomAccessFile implementation. Defaults to 5Mb.
constexpr size_t kReadAheadBytes = 5 * 1024 * 1024;
// The number of list object results
constexpr int kMaxRet = 1000;

void oss_initialize_with_throwable() {
  if (aos_http_io_initialize(nullptr, 0) != AOSE_OK) {
    throw std::exception();
  }
}

void oss_initialize(TF_Status* status) {
  static std::once_flag initFlag;
  try {
    std::call_once(initFlag, [] { oss_initialize_with_throwable(); });
  } catch (...) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "can not init OSS connection");
  }
  TF_SetStatus(status, TF_OK, "");
}

void oss_error_message(aos_status_s* status, std::string* msg) {
  *msg = status->req_id;
  if (aos_status_is_ok(status)) {
    return;
  }

  msg->append(" ");
  msg->append(std::to_string(status->code));

  if (status->code == 404) {
    msg->append(" object not exists!");
    return;
  }

  if (status->error_msg) {
    msg->append(" ");
    msg->append(status->error_msg);
    return;
  }
}

void ParseURI(const absl::string_view& fname, absl::string_view* scheme,
              absl::string_view* host, absl::string_view* path) {
  size_t scheme_chunk = fname.find("://");
  if (scheme_chunk == absl::string_view::npos) {
    return;
  }
  size_t host_chunk = fname.find("/", scheme_chunk + 3);
  if (host_chunk == absl::string_view::npos) {
    return;
  }
  *scheme = absl::string_view(fname).substr(0, scheme_chunk);
  *host = fname.substr(scheme_chunk + 3, host_chunk);
  *path = fname.substr(host_chunk, -1);
}

class OSSConnection {
 public:
  OSSConnection(const std::string& endPoint, const std::string& accessKey,
                const std::string& accessKeySecret) {
    aos_pool_create(&_pool, nullptr);
    _options = oss_request_options_create(_pool);
    _options->config = oss_config_create(_options->pool);
    aos_str_set(&_options->config->endpoint, endPoint.c_str());
    aos_str_set(&_options->config->access_key_id, accessKey.c_str());
    aos_str_set(&_options->config->access_key_secret, accessKeySecret.c_str());
    _options->config->is_cname = 0;
    _options->ctl = aos_http_controller_create(_options->pool, 0);
  }

  ~OSSConnection() {
    if (nullptr != _pool) {
      aos_pool_destroy(_pool);
    }
  }

  oss_request_options_t* getRequestOptions() { return _options; }
  aos_pool_t* getPool() { return _pool; }

 private:
  aos_pool_t* _pool = nullptr;
  oss_request_options_t* _options = nullptr;
};

class OSSRandomAccessFile {
 public:
  OSSRandomAccessFile(const std::string& endPoint, const std::string& accessKey,
                      const std::string& accessKeySecret,
                      const std::string& bucket, const std::string& object,
                      size_t kReadAheadBytes, size_t file_length)
      : shost(endPoint),
        sak(accessKey),
        ssk(accessKeySecret),
        sbucket(bucket),
        sobject(object),
        total_file_length_(file_length) {
    read_ahead_bytes_ = std::min(kReadAheadBytes, file_length);
  }

  int64_t Read(uint64_t offset, size_t n, char* buffer,
               TF_Status* status) const {
    if (n == 0) {
      TF_SetStatus(status, TF_OK, "");
      return 0;
    }

    // offset is 0 based, so last offset should be
    // just before total_file_length_
    if (offset >= total_file_length_) {
      std::string error_message =
          absl::StrCat("EOF reached, ", offset, " is read out of file length ",
                       total_file_length_);
      TF_SetStatus(status, TF_OUT_OF_RANGE, error_message.c_str());
      return 0;
    }

    if (offset + n > total_file_length_) {
      n = total_file_length_ - offset;
    }

    TF_VLog(1, "read %s from %d to %d", sobject.c_str(), offset, offset + n);

    absl::MutexLock lock(&mu_);
    const bool range_start_included = offset >= buffer_start_offset_;
    const bool range_end_included =
        offset + n <= buffer_start_offset_ + buffer_size_;
    if (range_start_included && range_end_included) {
      // The requested range can be filled from the buffer.
      const size_t offset_in_buffer =
          std::min<uint64_t>(offset - buffer_start_offset_, buffer_size_);
      const auto copy_size = std::min(n, buffer_size_ - offset_in_buffer);
      TF_VLog(1, "read from buffer ", offset_in_buffer, " to ",
              offset_in_buffer + copy_size, " total ", buffer_size_);
      std::copy(buffer_.begin() + offset_in_buffer,
                buffer_.begin() + offset_in_buffer + copy_size, buffer);
      TF_SetStatus(status, TF_OK, "");
      return copy_size;
    } else {
      // Update the buffer content based on the new requested range.
      const size_t desired_buffer_size =
          std::min(n + read_ahead_bytes_, total_file_length_);
      if (n > buffer_.capacity() ||
          desired_buffer_size > 2 * buffer_.capacity()) {
        // Re-allocate only if buffer capacity increased significantly.
        TF_VLog(1, "reserve buffer to ", desired_buffer_size);
        buffer_.reserve(desired_buffer_size);
      }

      buffer_start_offset_ = offset;
      TF_VLog(1, "load buffer", buffer_start_offset_);
      LoadBufferFromOSS(desired_buffer_size, status);
      if (TF_GetCode(status) != TF_OK) {
        return 0;
      }
      // Set the results.
      memcpy(buffer, buffer_.data(), std::min(buffer_size_, n));
      TF_SetStatus(status, TF_OK, "");
      return n;
    }
  }

 private:
  /// A helper function to actually read the data from OSS. This function loads
  /// buffer_ from OSS based on its current capacity.
  void LoadBufferFromOSS(size_t desired_buffer_size, TF_Status* status) const {
    size_t range_start = buffer_start_offset_;
    size_t range_end = buffer_start_offset_ + std::min(buffer_.capacity() - 1,
                                                       desired_buffer_size - 1);
    range_end = std::min(range_end, total_file_length_ - 1);

    OSSConnection conn(shost, sak, ssk);
    aos_pool_t* _pool = conn.getPool();
    oss_request_options_t* _options = conn.getRequestOptions();
    aos_string_t bucket_;
    aos_string_t object_;
    aos_table_t* headers_;
    aos_list_t tmp_buffer;
    aos_table_t* resp_headers;

    aos_list_init(&tmp_buffer);
    aos_str_set(&_options->config->endpoint, shost.c_str());
    aos_str_set(&_options->config->access_key_id, sak.c_str());
    aos_str_set(&_options->config->access_key_secret, ssk.c_str());
    _options->config->is_cname = 0;
    _options->ctl = aos_http_controller_create(_options->pool, 0);
    aos_str_set(&bucket_, sbucket.c_str());
    aos_str_set(&object_, sobject.c_str());
    headers_ = aos_table_make(_pool, 1);

    std::string range("bytes=");
    range.append(std::to_string(range_start))
        .append("-")
        .append(std::to_string(range_end));
    apr_table_set(headers_, "Range", range.c_str());
    TF_VLog(1, "read from OSS with ", range.c_str());

    aos_status_t* s =
        oss_get_object_to_buffer(_options, &bucket_, &object_, headers_,
                                 nullptr, &tmp_buffer, &resp_headers);

    if (!aos_status_is_ok(s)) {
      std::string msg;
      oss_error_message(s, &msg);
      TF_VLog(0, "read ", sobject, " failed, errMsg: ", msg);
      std::string error_message =
          absl::StrCat("read failed: ", sobject, " errMsg: ", msg);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    aos_buf_t* content = nullptr;
    int64_t size = 0;
    int64_t pos = 0;
    buffer_.clear();
    buffer_size_ = 0;

    // copy data to local buffer
    aos_list_for_each_entry(aos_buf_t, content, &tmp_buffer, node) {
      size = aos_buf_size(content);
      std::copy(content->pos, content->pos + size, buffer_.begin() + pos);
      pos += size;
    }
    buffer_size_ = pos;
    TF_SetStatus(status, TF_OK, "");
  }

  std::string shost;
  std::string sak;
  std::string ssk;
  std::string sbucket;
  std::string sobject;
  const size_t total_file_length_;
  size_t read_ahead_bytes_;

  mutable absl::Mutex mu_;
  mutable std::vector<char> buffer_ ABSL_GUARDED_BY(mu_);
  // The original file offset of the first byte in the buffer.
  mutable size_t buffer_start_offset_ ABSL_GUARDED_BY(mu_) = 0;
  mutable size_t buffer_size_ ABSL_GUARDED_BY(mu_) = 0;
};

class OSSWritableFile {
 public:
  OSSWritableFile(const std::string& endPoint, const std::string& accessKey,
                  const std::string& accessKeySecret, const std::string& bucket,
                  const std::string& object, size_t part_size)
      : shost(endPoint),
        sak(accessKey),
        ssk(accessKeySecret),
        sbucket(bucket),
        sobject(object),
        part_size_(part_size),
        is_closed_(false),
        part_number_(1) {
    InitAprPool();
  }

  ~OSSWritableFile() { ReleaseAprPool(); }

  void Append(const char* buffer, size_t n, TF_Status* status) {
    absl::MutexLock lock(&mu_);
    CheckClosed(status);
    if (TF_GetCode(status) != TF_OK) {
      return;
    }
    InitAprPool();
    if (CurrentBufferLength() >= part_size_) {
      FlushInternal(status);
      if (TF_GetCode(status) != TF_OK) {
        return;
      }
    }

    aos_buf_t* tmp_buf = aos_create_buf(pool_, n + 1);
    aos_buf_append_string(pool_, tmp_buf, buffer, n);
    aos_list_add_tail(&tmp_buf->node, &buffer_);
    TF_SetStatus(status, TF_OK, "");
  }

  void Close(TF_Status* status) {
    absl::MutexLock lock(&mu_);
    CheckClosed(status);
    if (TF_GetCode(status) != TF_OK) {
      return;
    }
    InitAprPool();
    FlushInternal(status);
    if (TF_GetCode(status) != TF_OK) {
      return;
    }
    aos_table_t* complete_headers = nullptr;
    aos_table_t* resp_headers = nullptr;
    aos_status_t* aos_status = nullptr;
    oss_list_upload_part_params_t* params = nullptr;
    aos_list_t complete_part_list;
    oss_list_part_content_t* part_content = nullptr;
    oss_complete_part_content_t* complete_part_content = nullptr;
    aos_string_t upload_id;
    aos_str_set(&upload_id, upload_id_.c_str());

    params = oss_create_list_upload_part_params(pool_);
    aos_list_init(&complete_part_list);
    aos_status = oss_list_upload_part(options_, &bucket_, &object_, &upload_id,
                                      params, &resp_headers);

    if (!aos_status_is_ok(aos_status)) {
      std::string msg;
      oss_error_message(aos_status, &msg);
      TF_VLog(0, "List multipart ", sobject, " failed, errMsg: ", msg);
      std::string error_message =
          absl::StrCat("List multipart failed: ", sobject, " errMsg: ", msg);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    aos_list_for_each_entry(oss_list_part_content_t, part_content,
                            &params->part_list, node) {
      complete_part_content = oss_create_complete_part_content(pool_);
      aos_str_set(&complete_part_content->part_number,
                  part_content->part_number.data);
      aos_str_set(&complete_part_content->etag, part_content->etag.data);
      aos_list_add_tail(&complete_part_content->node, &complete_part_list);
    }

    aos_status = oss_complete_multipart_upload(options_, &bucket_, &object_,
                                               &upload_id, &complete_part_list,
                                               complete_headers, &resp_headers);

    if (!aos_status_is_ok(aos_status)) {
      std::string msg;
      oss_error_message(aos_status, &msg);
      TF_VLog(0, "Complete multipart ", sobject, " failed, errMsg: ", msg);
      std::string error_message = absl::StrCat(
          "Complete multipart failed: ", sobject, " errMsg: ", msg);
      TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
      return;
    }

    is_closed_ = true;
    TF_SetStatus(status, TF_OK, "");
  }

  void Flush(TF_Status* status) {
    absl::MutexLock lock(&mu_);
    CheckClosed(status);
    if (TF_GetCode(status) != TF_OK) {
      return;
    }
    if (CurrentBufferLength() >= part_size_) {
      InitAprPool();
      FlushInternal(status);
      if (TF_GetCode(status) != TF_OK) {
        return;
      }
    }
    TF_SetStatus(status, TF_OK, "");
  }

  void Sync(TF_Status* status) { Flush(status); }

 private:
  void InitAprPool() {
    if (nullptr != pool_) {
      return;
    }
    aos_pool_create(&pool_, nullptr);
    options_ = oss_request_options_create(pool_);
    options_->config = oss_config_create(options_->pool);
    aos_str_set(&options_->config->endpoint, shost.c_str());
    aos_str_set(&options_->config->access_key_id, sak.c_str());
    aos_str_set(&options_->config->access_key_secret, ssk.c_str());
    options_->config->is_cname = 0;
    options_->ctl = aos_http_controller_create(options_->pool, 0);

    aos_str_set(&bucket_, sbucket.c_str());
    aos_str_set(&object_, sobject.c_str());

    headers_ = aos_table_make(pool_, 1);
    aos_list_init(&buffer_);
  }

  void ReleaseAprPool() {
    if (nullptr != pool_) {
      aos_pool_destroy(pool_);
      pool_ = nullptr;
    }
  }

  void InitMultiUpload(TF_Status* status) {
    if (!upload_id_.empty()) {
      TF_SetStatus(status, TF_OK, "");
      return;
    }

    aos_string_t uploadId;
    aos_status_t* aos_status = nullptr;
    aos_table_t* resp_headers = nullptr;

    InitAprPool();
    aos_status = oss_init_multipart_upload(options_, &bucket_, &object_,
                                           &uploadId, headers_, &resp_headers);

    if (!aos_status_is_ok(aos_status)) {
      std::string msg;
      oss_error_message(aos_status, &msg);
      TF_VLog(0, "Init multipart upload ", sobject, " failed, errMsg: ", msg);
      std::string error_message = absl::StrCat(
          "Init multipart upload failed: ", sobject, " errMsg: ", msg);
      TF_SetStatus(status, TF_UNAVAILABLE, error_message.c_str());
      return;
    }

    upload_id_ = uploadId.data;
    TF_SetStatus(status, TF_OK, "");
  }

  void FlushInternal(TF_Status* status) {
    aos_table_t* resp_headers = nullptr;
    aos_status_s* aos_status = nullptr;
    aos_string_t uploadId;
    if (CurrentBufferLength() > 0) {
      InitMultiUpload(status);
      if (TF_GetCode(status) != TF_OK) {
        return;
      }
      aos_str_set(&uploadId, upload_id_.c_str());
      aos_status =
          oss_upload_part_from_buffer(options_, &bucket_, &object_, &uploadId,
                                      part_number_, &buffer_, &resp_headers);

      if (!aos_status_is_ok(aos_status)) {
        std::string msg;
        oss_error_message(aos_status, &msg);
        TF_VLog(0, "Upload multipart ", sobject, " failed, errMsg: ", msg);
        std::string error_message = absl::StrCat(
            "Upload multipart failed: ", sobject, " errMsg: ", msg);
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }

      TF_VLog(1, " upload ", sobject, " with part", part_number_, " succ");
      part_number_++;
      ReleaseAprPool();
      InitAprPool();
    }
    TF_SetStatus(status, TF_OK, "");
  }

  const size_t CurrentBufferLength() { return aos_buf_list_len(&buffer_); }

  void CheckClosed(TF_Status* status) {
    if (is_closed_) {
      TF_SetStatus(status, TF_INTERNAL, "Already closed.");
      return;
    }
    TF_SetStatus(status, TF_OK, "");
  }

  std::string shost;
  std::string sak;
  std::string ssk;
  std::string sbucket;
  std::string sobject;
  size_t part_size_;

  aos_pool_t* pool_ = nullptr;
  oss_request_options_t* options_ = nullptr;
  aos_string_t bucket_;
  aos_string_t object_;
  aos_table_t* headers_ = nullptr;
  aos_list_t buffer_;
  std::string upload_id_;

  bool is_closed_;
  absl::Mutex mu_;
  int64_t part_number_;
};

namespace tf_random_access_file {
void Cleanup(TF_RandomAccessFile* file) {
  auto oss_file = static_cast<OSSRandomAccessFile*>(file->plugin_file);
  delete oss_file;
}

int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
             char* buffer, TF_Status* status) {
  auto oss_file = static_cast<OSSRandomAccessFile*>(file->plugin_file);
  return oss_file->Read(offset, n, buffer, status);
}

}  // namespace tf_random_access_file

namespace tf_writable_file {

static void Cleanup(TF_WritableFile* file) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  delete oss_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  oss_file->Append(buffer, n, status);
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Stat not implemented");
  return -1;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  oss_file->Flush(status);
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  oss_file->Sync(status);
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  auto oss_file = static_cast<OSSWritableFile*>(file->plugin_file);
  oss_file->Close(status);
}

}  // namespace tf_writable_file

namespace tf_read_only_memory_region {
void Cleanup(TF_ReadOnlyMemoryRegion* region) {}

const void* Data(const TF_ReadOnlyMemoryRegion* region) { return nullptr; }

uint64_t Length(const TF_ReadOnlyMemoryRegion* region) { return 0; }

}  // namespace tf_read_only_memory_region

namespace tf_oss_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

// Splits a oss path to endpoint bucket object and token
// For example
// "oss://bucket-name\x01id=accessid\x02key=accesskey\x02host=endpoint/path/to/file.txt"
void ParseOSSURIPath(const absl::string_view fname, std::string& bucket,
                     std::string& object, std::string& host,
                     std::string& access_id, std::string& access_key,
                     TF_Status* status) {
  absl::string_view scheme, bucketp, remaining;
  ParseURI(fname, &scheme, &bucketp, &remaining);

  if (scheme != "oss") {
    TF_SetStatus(
        status, TF_INTERNAL,
        absl::StrCat("OSS path does not start with 'oss://':", fname).c_str());
    return;
  }

  absl::ConsumePrefix(&remaining, kDelim);
  object = std::string(remaining);

  std::string bucketDelim = "?";
  std::string accessDelim = "&";
  if (bucketp.find('\x01') != absl::string_view::npos) {
    bucketDelim = "\x01";
    accessDelim = "\x02";
  }

  // contains id, key, host information
  size_t pos = bucketp.find(bucketDelim);
  bucket = std::string(bucketp.substr(0, pos));
  absl::string_view access_info = bucketp.substr(pos + 1);
  std::vector<std::string> access_infos =
      absl::StrSplit(access_info, accessDelim);
  for (const auto& key_value : access_infos) {
    absl::string_view data(key_value);
    size_t pos = data.find('=');
    if (pos == absl::string_view::npos) {
      TF_SetStatus(status, TF_INTERNAL,
                   absl::StrCat("OSS path access info faied: ", fname,
                                " info:", key_value)
                       .c_str());
      return;
    }
    absl::string_view key = data.substr(0, pos);
    absl::string_view value = data.substr(pos + 1);
    if (absl::StartsWith(key, kOSSAccessIdKey)) {
      access_id = std::string(value);
    } else if (absl::StartsWith(key, kOSSAccessKeyKey)) {
      access_key = std::string(value);
    } else if (absl::StartsWith(key, kOSSHostKey)) {
      host = std::string(value);
    } else {
      TF_SetStatus(status, TF_INTERNAL,
                   absl::StrCat("OSS path access info faied: ", fname,
                                " unkown info:", key_value)
                       .c_str());
      return;
    }
  }

  if (bucket.empty()) {
    TF_SetStatus(status, TF_INTERNAL,
                 absl::StrCat("OSS path does not contain a bucket name:", fname)
                     .c_str());
    return;
  }

  if (access_id.empty() || access_key.empty() || host.empty()) {
    TF_SetStatus(
        status, TF_INTERNAL,
        absl::StrCat("OSS path does not contain valid access info:", fname)
            .c_str());
    return;
  }

  TF_VLog(1, "bucket: ", bucket, ",access_id: ", access_id,
          ",access_key: ", access_key, ",host: ", host);

  TF_SetStatus(status, TF_OK, "");
}

void RetrieveObjectMetadata(aos_pool_t* pool,
                            const oss_request_options_t* options,
                            const std::string& bucket,
                            const std::string& object, TF_FileStatistics* stat,
                            TF_Status* status) {
  aos_string_t oss_bucket;
  aos_string_t oss_object;
  aos_table_t* headers = nullptr;
  aos_table_t* resp_headers = nullptr;
  aos_status_t* aos_status = nullptr;
  char* content_length_str = nullptr;
  char* object_date_str = nullptr;

  if (object.empty()) {  // root always exists
    stat->is_directory = true;
    stat->length = 0;
    TF_SetStatus(status, TF_OK, "");
    return;
  }

  aos_str_set(&oss_bucket, bucket.c_str());
  aos_str_set(&oss_object, object.c_str());
  headers = aos_table_make(pool, 0);

  aos_status = oss_head_object(options, &oss_bucket, &oss_object, headers,
                               &resp_headers);
  if (aos_status_is_ok(aos_status)) {
    content_length_str = (char*)apr_table_get(resp_headers, OSS_CONTENT_LENGTH);
    if (content_length_str != nullptr) {
      stat->length = static_cast<int64_t>(atoll(content_length_str));
      TF_VLog(1, "_RetrieveObjectMetadata object: ", object,
              " , with length: ", stat->length);
    }

    object_date_str = (char*)apr_table_get(resp_headers, OSS_DATE);
    if (object_date_str != nullptr) {
      // the time is GMT Date, format like below
      // Date: Fri, 24 Feb 2012 07:32:52 GMT
      std::tm tm = {};
      strptime(object_date_str, "%a, %d %b %Y %H:%M:%S", &tm);
      stat->mtime_nsec = static_cast<int64_t>(mktime(&tm) * 1000) * 1e9;

      TF_VLog(1, "_RetrieveObjectMetadata object: ", object,
              " , with time: ", stat->mtime_nsec);
    } else {
      TF_VLog(0, "find ", object, " with no datestr");
      std::string error_message =
          absl::StrCat("find", object, " with no datestr");
      TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
      return;
    }

    if (object[object.length() - 1] == '/') {
      stat->is_directory = true;
    } else {
      stat->is_directory = false;
    }

    TF_SetStatus(status, TF_OK, "");
    return;
  } else {
    std::string msg;
    oss_error_message(aos_status, &msg);
    TF_VLog(1, "can not find object: ", object, ", with bucket: ", bucket,
            ", errMsg: ", msg);
    std::string error_message =
        absl::StrCat("can not find ", object, " errMsg: ", msg);
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  std::string object, bucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(path, bucket, object, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  TF_FileStatistics stat;
  OSSConnection conn(host, access_id, access_key);
  RetrieveObjectMetadata(conn.getPool(), conn.getRequestOptions(), bucket,
                         object, &stat, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  file->plugin_file =
      new OSSRandomAccessFile(host, access_id, access_key, bucket, object,
                              kReadAheadBytes, stat.length);
  TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  std::string object, bucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(path, bucket, object, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  file->plugin_file = new OSSWritableFile(host, access_id, access_key, bucket,
                                          object, kUploadPartBytes);
  TF_SetStatus(status, TF_OK, "");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "NewAppendableFile not implemented");
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "NewReadOnlyMemoryRegionFromFile not implemented");
}

// For GetChildren , we should not return prefix
void ListObjects(aos_pool_t* pool, const oss_request_options_t* options,
                 const std::string& bucket, const std::string& key,
                 std::vector<std::string>* result, bool return_all,
                 bool return_full_path, bool should_remove_suffix,
                 int max_ret_per_iterator, TF_Status* status) {
  aos_string_t bucket_;
  aos_status_t* s = nullptr;
  oss_list_object_params_t* params = nullptr;
  oss_list_object_content_t* content = nullptr;
  const char* next_marker = "";

  aos_str_set(&bucket_, bucket.c_str());
  params = oss_create_list_object_params(pool);
  params->max_ret = max_ret_per_iterator;
  aos_str_set(&params->prefix, key.c_str());
  aos_str_set(&params->marker, next_marker);

  do {
    s = oss_list_object(options, &bucket_, params, nullptr);
    if (!aos_status_is_ok(s)) {
      std::string msg;
      oss_error_message(s, &msg);
      TF_VLog(0, "cam not list object ", key, " errMsg: ", msg);
      std::string error_message =
          absl::StrCat("can not list object:", key, " errMsg: ", msg);
      TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
      return;
    }

    aos_list_for_each_entry(oss_list_object_content_t, content,
                            &params->object_list, node) {
      int path_length = content->key.len;
      if (should_remove_suffix && path_length > 0 &&
          content->key.data[content->key.len - 1] == '/') {
        path_length = content->key.len - 1;
      }
      if (return_full_path) {
        std::string child(content->key.data, 0, path_length);
        result->push_back(child);
      } else {
        int prefix_len = (key.length() > 0 && key.at(key.length() - 1) != '/')
                             ? key.length() + 1
                             : key.length();
        // remove prefix for GetChildren
        if (content->key.len > prefix_len) {
          std::string child(content->key.data + prefix_len, 0,
                            path_length - prefix_len);
          result->push_back(child);
        }
      }
    }

    next_marker = apr_psprintf(pool, "%.*s", params->next_marker.len,
                               params->next_marker.data);

    aos_str_set(&params->marker, next_marker);
    aos_list_init(&params->object_list);
    aos_list_init(&params->common_prefix_list);
  } while (params->truncated == AOS_TRUE && return_all);

  TF_SetStatus(status, TF_OK, "");
}

void StatInternal(aos_pool_t* pool, const oss_request_options_t* options,
                  const std::string& bucket, const std::string& object,
                  TF_FileStatistics* stat, TF_Status* status) {
  RetrieveObjectMetadata(pool, options, bucket, object, stat, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  // add suffix
  std::string objectName = object + kDelim;
  RetrieveObjectMetadata(pool, options, bucket, objectName, stat, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_VLog(1, "RetrieveObjectMetadata for object: ", objectName,
            " directory success");
    stat->is_directory = true;
    TF_SetStatus(status, TF_OK, "");
    return;
  }

  // check list if it has children
  std::vector<std::string> listing;
  ListObjects(pool, options, bucket, object, &listing, true, false, false, 10,
              status);

  if (TF_GetCode(status) == TF_OK && !listing.empty()) {
    if (absl::EndsWith(object, "/")) {
      stat->is_directory = true;
    }
    stat->length = 0;
    TF_VLog(1, "RetrieveObjectMetadata for object: ", object,
            " get children success");
    TF_SetStatus(status, TF_OK, "");
    return;
  }

  TF_VLog(1, "_StatInternal for object: ", object,
          ", failed with bucket: ", bucket);
  std::string error_message = absl::StrCat("can not find ", object);
  TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  std::string object, bucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(path, bucket, object, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  StatInternal(pool, ossOptions, bucket, object, stats, status);
}

void CreateDirInternal(aos_pool_t* pool, const oss_request_options_t* options,
                       const std::string& bucket, const std::string& dirname,
                       TF_Status* status) {
  TF_FileStatistics stat;
  RetrieveObjectMetadata(pool, options, bucket, dirname, &stat, status);
  if (TF_GetCode(status) == TF_OK) {
    if (!stat.is_directory) {
      TF_VLog(0, "object already exists as a file: ", dirname);
      std::string error_message =
          absl::StrCat("object already exists as a file: ", dirname);
      TF_SetStatus(status, TF_ALREADY_EXISTS, error_message.c_str());
      return;
    } else {
      TF_SetStatus(status, TF_OK, "");
      return;
    }
  }

  std::string object = dirname;
  if (dirname.at(dirname.length() - 1) != '/') {
    object += '/';
  }

  aos_status_t* s;
  aos_table_t* headers;
  aos_table_t* resp_headers;
  aos_string_t bucket_;
  aos_string_t object_;
  const char* data = "";
  aos_list_t buffer;
  aos_buf_t* content;

  aos_str_set(&bucket_, bucket.c_str());
  aos_str_set(&object_, object.c_str());
  headers = aos_table_make(pool, 0);

  aos_list_init(&buffer);
  content = aos_buf_pack(options->pool, data, strlen(data));
  aos_list_add_tail(&content->node, &buffer);
  s = oss_put_object_from_buffer(options, &bucket_, &object_, &buffer, headers,
                                 &resp_headers);

  if (!aos_status_is_ok(s)) {
    std::string msg;
    oss_error_message(s, &msg);
    TF_VLog(1, "mkdir ", dirname, " failed, errMsg: ", msg);
    std::string error_message =
        absl::StrCat("mkdir failed: ", dirname, " errMsg: ", msg);
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  std::string object, bucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(path, bucket, object, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  absl::string_view dirs(object);
  std::vector<std::string> splitPaths =
      absl::StrSplit(dirs, '/', absl::SkipEmpty());
  if (splitPaths.size() < 2) {
    CreateDirInternal(pool, ossOptions, bucket, object, status);
    return;
  }

  TF_FileStatistics stat;
  auto BaseName = [](const std::string& name) {
    return name.substr(name.find_last_of('/') + 1);
  };
  std::string parent = BaseName(object);
  StatInternal(pool, ossOptions, bucket, parent, &stat, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_VLog(0, "CreateDir() failed with bucket: ", bucket,
            ", parent: ", parent);
    std::string error_message =
      absl::StrCat("parent does not exists: ", parent);
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  if (!stat.is_directory) {
    std::string error_message =
      absl::StrCat("can not mkdir because parent is a file: ", parent);
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  CreateDirInternal(pool, ossOptions, bucket, object, status);
}

void DeleteObjectInternal(const oss_request_options_t* options,
                          const std::string& bucket, const std::string& object,
                          TF_Status* status) {
  aos_string_t bucket_;
  aos_string_t object_;
  aos_table_t* resp_headers = nullptr;
  aos_status_t* s = nullptr;

  aos_str_set(&bucket_, bucket.c_str());
  aos_str_set(&object_, object.c_str());

  s = oss_delete_object(options, &bucket_, &object_, &resp_headers);
  if (!aos_status_is_ok(s)) {
    std::string msg;
    oss_error_message(s, &msg);
    TF_VLog(0, "delete ", object, " failed, errMsg: ", msg);
    std::string error_message =
        absl::StrCat("delete failed: ", object, " errMsg: ", msg);
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  TF_SetStatus(status, TF_OK, "");
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  std::string object, bucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(path, bucket, object, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  std::vector<std::string> children;
  ListObjects(pool, oss_options, bucket, object, &children, true, false, false,
              10, status);
  if (TF_GetCode(status) == TF_OK && !children.empty()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Cannot delete a non-empty directory.");
    return;
  }

  DeleteObjectInternal(oss_options, bucket, object, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  // Maybe should add slash
  DeleteObjectInternal(oss_options, bucket, object.append(kDelim), status);
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  std::string object, bucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(path, bucket, object, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  DeleteObjectInternal(oss_options, bucket, object, status);
}

// TODO: Better implementation
void IsDirectory(const std::string& fname, TF_Status* status) {
  TF_FileStatistics stat;
  Stat(nullptr, fname.c_str(), &stat, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }
  if (!stat.is_directory) {
    std::string error_message = absl::StrCat(fname + " is not a directory");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

aos_status_t* RenameFileInternal(const oss_request_options_t* oss_options,
                                 aos_pool_t* pool,
                                 const aos_string_t& source_bucket,
                                 const aos_string_t& source_object,
                                 const aos_string_t& dest_bucket,
                                 const aos_string_t& dest_object,
                                 TF_Status* status) {
  aos_status_t* resp_status;
  aos_table_t* resp_headers;
  aos_table_t* headers = aos_table_make(pool, 0);
  aos_string_t upload_id;

  oss_list_upload_part_params_t* list_upload_part_params;
  oss_upload_part_copy_params_t* upload_part_copy_params =
      oss_create_upload_part_copy_params(pool);
  oss_list_part_content_t* part_content;
  aos_list_t complete_part_list;
  oss_complete_part_content_t* complete_content;
  aos_table_t* list_part_resp_headers = nullptr;
  aos_table_t* complete_resp_headers = nullptr;

  // get file size
  TF_FileStatistics stat;
  StatInternal(pool, oss_options, std::string(source_bucket.data),
               std::string(source_object.data), &stat, status);
  uint64_t file_size = stat.length;

  // file size bigger than kUploadPartBytes, need to split into multi parts
  if (file_size > kUploadPartBytes) {
    resp_status =
        oss_init_multipart_upload(oss_options, &dest_bucket, &dest_object,
                                  &upload_id, headers, &resp_headers);
    if (aos_status_is_ok(resp_status)) {
      TF_VLog(1, "init multipart upload succeeded, upload_id is %s",
              upload_id.data);
    } else {
      return resp_status;
    }

    // process for each single part
    int parts = ceil(double(file_size) / double(kUploadPartBytes));
    for (int i = 0; i < parts - 1; i++) {
      int64_t range_start = i * kUploadPartBytes;
      int64_t range_end = (i + 1) * kUploadPartBytes - 1;
      int part_num = i + 1;

      aos_str_set(&upload_part_copy_params->source_bucket, source_bucket.data);
      aos_str_set(&upload_part_copy_params->source_object, source_object.data);
      aos_str_set(&upload_part_copy_params->dest_bucket, dest_bucket.data);
      aos_str_set(&upload_part_copy_params->dest_object, dest_object.data);
      aos_str_set(&upload_part_copy_params->upload_id, upload_id.data);

      upload_part_copy_params->part_num = part_num;
      upload_part_copy_params->range_start = range_start;
      upload_part_copy_params->range_end = range_end;

      headers = aos_table_make(pool, 0);

      resp_status = oss_upload_part_copy(oss_options, upload_part_copy_params,
                                         headers, &resp_headers);
      if (aos_status_is_ok(resp_status)) {
        TF_VLog(1, "upload part ", part_num, " copy succeeded");
      } else {
        return resp_status;
      }
    }

    int64_t range_start = (parts - 1) * kUploadPartBytes;
    int64_t range_end = file_size - 1;

    aos_str_set(&upload_part_copy_params->source_bucket, source_bucket.data);
    aos_str_set(&upload_part_copy_params->source_object, source_object.data);
    aos_str_set(&upload_part_copy_params->dest_bucket, dest_bucket.data);
    aos_str_set(&upload_part_copy_params->dest_object, dest_object.data);
    aos_str_set(&upload_part_copy_params->upload_id, upload_id.data);
    upload_part_copy_params->part_num = parts;
    upload_part_copy_params->range_start = range_start;
    upload_part_copy_params->range_end = range_end;

    headers = aos_table_make(pool, 0);

    resp_status = oss_upload_part_copy(oss_options, upload_part_copy_params,
                                       headers, &resp_headers);
    if (aos_status_is_ok(resp_status)) {
      TF_VLog(1, "upload part ", parts, " copy succeeded");
    } else {
      return resp_status;
    }

    headers = aos_table_make(pool, 0);
    list_upload_part_params = oss_create_list_upload_part_params(pool);
    list_upload_part_params->max_ret = kMaxRet;
    aos_list_init(&complete_part_list);
    resp_status = oss_list_upload_part(oss_options, &dest_bucket, &dest_object,
                                       &upload_id, list_upload_part_params,
                                       &list_part_resp_headers);
    aos_list_for_each_entry(oss_list_part_content_t, part_content,
                            &list_upload_part_params->part_list, node) {
      complete_content = oss_create_complete_part_content(pool);
      aos_str_set(&complete_content->part_number,
                  part_content->part_number.data);
      aos_str_set(&complete_content->etag, part_content->etag.data);
      aos_list_add_tail(&complete_content->node, &complete_part_list);
    }

    resp_status = oss_complete_multipart_upload(
        oss_options, &dest_bucket, &dest_object, &upload_id,
        &complete_part_list, headers, &complete_resp_headers);
    if (aos_status_is_ok(resp_status)) {
      TF_VLog(1, "complete multipart upload succeeded");
    }
  } else {
    resp_status =
        oss_copy_object(oss_options, &source_bucket, &source_object,
                        &dest_bucket, &dest_object, headers, &resp_headers);
  }
  return resp_status;
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  std::string sobject, sbucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(src, sbucket, sobject, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  std::string dobject, dbucket;
  std::string dhost, daccess_id, daccess_key;
  ParseOSSURIPath(dst, dbucket, dobject, dhost, daccess_id, daccess_key,
                  status);
  if (TF_GetCode(status) != TF_OK) {
    return;
  }

  if (host != dhost || access_id != daccess_id || access_key != daccess_key) {
    TF_VLog(0, "rename ", src, " to ", dst, " failed, with errMsg: ",
            " source oss cluster does not match dest oss cluster");
    std::string error_message =
        absl::StrCat("rename ", src, " to ", dst, " failed, errMsg: ",
                     "source oss cluster does not match dest oss cluster");
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();

  aos_status_t* resp_status;
  aos_string_t source_bucket;
  aos_string_t source_object;
  aos_string_t dest_bucket;
  aos_string_t dest_object;

  aos_str_set(&source_bucket, sbucket.c_str());
  aos_str_set(&dest_bucket, dbucket.c_str());

  IsDirectory(src, status);
  if (TF_GetCode(status) == TF_OK) {
    if (!absl::EndsWith(sobject, "/")) {
      sobject += "/";
    }
    if (!absl::EndsWith(dobject, "/")) {
      dobject += "/";
    }
    std::vector<std::string> childPaths;
    ListObjects(pool, oss_options, sbucket, sobject, &childPaths, true, false,
                false, kMaxRet, status);
    if (TF_GetCode(status) != TF_OK) {
      return;
    }

    for (const auto& child : childPaths) {
      std::string tmp_sobject = sobject + child;
      std::string tmp_dobject = dobject + child;

      aos_str_set(&source_object, tmp_sobject.c_str());
      aos_str_set(&dest_object, tmp_dobject.c_str());

      resp_status =
          RenameFileInternal(oss_options, pool, source_bucket, source_object,
                             dest_bucket, dest_object, status);
      if (!aos_status_is_ok(resp_status)) {
        std::string msg;
        oss_error_message(resp_status, &msg);
        TF_VLog(0, "rename ", src, " to ", dst,
                " failed, with specific file:  ", tmp_sobject,
                ", with errMsg: ", msg);
        std::string error_message =
            absl::StrCat("rename ", src, " to ", dst, " failed, errMsg: ", msg);
        TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
        return;
      }
      DeleteObjectInternal(oss_options, sbucket, tmp_sobject, status);
    }
  }

  aos_str_set(&source_object, sobject.c_str());
  aos_str_set(&dest_object, dobject.c_str());
  resp_status =
      RenameFileInternal(oss_options, pool, source_bucket, source_object,
                         dest_bucket, dest_object, status);
  if (!aos_status_is_ok(resp_status)) {
    std::string msg;
    oss_error_message(resp_status, &msg);
    TF_VLog(0, "rename ", src, " to ", dst, " failed, errMsg: ", msg);
    std::string error_message =
        absl::StrCat("rename ", src, " to ", dst, " failed, errMsg: ", msg);
    TF_SetStatus(status, TF_INTERNAL, error_message.c_str());
    return;
  }

  DeleteObjectInternal(oss_options, sbucket, sobject, status);
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  TF_FileStatistics stat;
  Stat(filesystem, path, &stat, status);
  if (TF_GetCode(status) != TF_OK) {
    std::string error_message = absl::StrCat(path, " does not exists");
    TF_SetStatus(status, TF_NOT_FOUND, error_message.c_str());
  }
}

static int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                           TF_Status* status) {
  TF_FileStatistics stat;
  Stat(filesystem, path, &stat, status);
  if (TF_GetCode(status) != TF_OK) {
    return 0;
  }

  TF_SetStatus(status, TF_OK, "");
  return stat.length;
}

static int GetChildren(const TF_Filesystem* filesystem, const char* path,
                       char*** entries, TF_Status* status) {
  oss_initialize(status);
  if (TF_GetCode(status) != TF_OK) {
    return 0;
  }
  std::string object, bucket;
  std::string host, access_id, access_key;
  ParseOSSURIPath(path, bucket, object, host, access_id, access_key, status);
  if (TF_GetCode(status) != TF_OK) {
    return 0;
  }

  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  std::vector<std::string> result;
  ListObjects(pool, oss_options, bucket, object, &result, true, false, true,
              kMaxRet, status);
  if (TF_GetCode(status) != TF_OK) {
    return 0;
  }

  int num_entries = result.size();
  *entries = static_cast<char**>(
      plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
  for (int i = 0; i < num_entries; i++) {
    (*entries)[i] = static_cast<char*>(
        plugin_memory_allocate(strlen(result[i].c_str()) + 1));
    memcpy((*entries)[i], result[i].c_str(), strlen(result[i].c_str()) + 1);
  }
  TF_SetStatus(status, TF_OK, "");
  return num_entries;
}

}  // namespace tf_oss_filesystem

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
      plugin_memory_allocate(TF_RANDOM_ACCESS_FILE_OPS_SIZE));
  ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
  ops->random_access_file_ops->read = tf_random_access_file::Read;

  ops->writable_file_ops = static_cast<TF_WritableFileOps*>(
      plugin_memory_allocate(TF_WRITABLE_FILE_OPS_SIZE));
  ops->writable_file_ops->cleanup = tf_writable_file::Cleanup;
  ops->writable_file_ops->append = tf_writable_file::Append;
  ops->writable_file_ops->tell = tf_writable_file::Tell;
  ops->writable_file_ops->flush = tf_writable_file::Flush;
  ops->writable_file_ops->sync = tf_writable_file::Sync;
  ops->writable_file_ops->close = tf_writable_file::Close;

  ops->read_only_memory_region_ops = static_cast<TF_ReadOnlyMemoryRegionOps*>(
      plugin_memory_allocate(TF_READ_ONLY_MEMORY_REGION_OPS_SIZE));
  ops->read_only_memory_region_ops->cleanup =
      tf_read_only_memory_region::Cleanup;
  ops->read_only_memory_region_ops->data = tf_read_only_memory_region::Data;
  ops->read_only_memory_region_ops->length = tf_read_only_memory_region::Length;

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_oss_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_oss_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_oss_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_oss_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_oss_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_oss_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_oss_filesystem::CreateDir;
  ops->filesystem_ops->delete_file = tf_oss_filesystem::DeleteFile;
  ops->filesystem_ops->delete_dir = tf_oss_filesystem::DeleteDir;
  ops->filesystem_ops->copy_file = tf_oss_filesystem::CopyFile;
  ops->filesystem_ops->rename_file = tf_oss_filesystem::RenameFile;
  ops->filesystem_ops->path_exists = tf_oss_filesystem::PathExists;
  ops->filesystem_ops->get_file_size = tf_oss_filesystem::GetFileSize;
  ops->filesystem_ops->stat = tf_oss_filesystem::Stat;
  ops->filesystem_ops->get_children = tf_oss_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_oss_filesystem::TranslateName;
}

}  // namespace oss
}  // namespace io
}  // namespace tensorflow
