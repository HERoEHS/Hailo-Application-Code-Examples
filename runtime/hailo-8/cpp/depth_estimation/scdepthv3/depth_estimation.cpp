/**
 * Copyright 2021 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/
/**
 * @ file semseg
 * This example demonstrates using virtual streams over c++
 **/

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <chrono>
#include <thread>
#include <atomic>

using hailort::Device;
using hailort::Hef;
using hailort::Expected;
using hailort::make_unexpected;
using hailort::ConfiguredNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::InputVStream;
using hailort::OutputVStream;
using hailort::MemoryView;

std::string getCmdOption(int argc, char *argv[], const std::string &longOption, const std::string &shortOption) {
    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find(longOption) == 0 || arg.find(shortOption) == 0) {
            std::size_t found = arg.find("=") + 1;
            cmd = arg.substr(found, 200);  
            return cmd;
        }
    }
    return cmd;
}

Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(Device &device, const std::string &hef_file) {
    auto hef = Hef::create(hef_file);
    if (!hef) {
        return make_unexpected(hef.status());
    }

    auto configure_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!configure_params) {
        return make_unexpected(configure_params.status());
    }

    auto network_groups = device.configure(hef.value(), configure_params.value());
    if (!network_groups) {
        return make_unexpected(network_groups.status());
    }

    if (1 != network_groups->size()) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}

template <typename T> hailo_status write_all(std::vector<InputVStream> &input, cv::VideoCapture &capture,
                                            int height, int width, int channels, std::atomic<bool> &should_stop) {
    std::cout << "-I- Started write thread" << std::endl;
    cv::Mat frame;
    while (!should_stop) {
        if (!capture.read(frame)) {
            std::cout << "Reached end of video stream, stopping." << std::endl;
            should_stop = true;
            break;
        }

        if (frame.channels() == 3)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        if (frame.rows != height || frame.cols != width)
            cv::resize(frame, frame, cv::Size(width, height), cv::INTER_AREA);

        int factor = std::is_same<T, uint8_t>::value ? 1 : 4;  // In case we use float32_t, we have 4 bytes per component
        auto status = input[0].write(MemoryView(frame.data, height * width * channels * factor));
        if (HAILO_SUCCESS != status) {
            should_stop = true;
            return status;
        }
    }
    std::cout << "-I- Finished write thread" << std::endl;
    return HAILO_SUCCESS;
}

template <typename T> cv::Mat scdepth_post_process(std::vector<T>& logits, int height, int width) {
    double min;
    double max;
    
    cv::Mat output(height, width, CV_32F, cv::Scalar(0));
    cv::Mat input(height, width, CV_32F, logits.data());

    cv::exp(-input, output);
    output = 1 / (1 + output);
    output = 1 / (output * 10 + 0.009);
    
    cv::minMaxIdx(output, &min, &max);
    output.convertTo(output, CV_8U, 255 / (max-min), -min);
    cv::applyColorMap(output, output, cv::COLORMAP_PLASMA);

    return output;
}

template <typename T> hailo_status read_all(std::vector<OutputVStream> &output, int height, int width, std::atomic<bool> &should_stop) {
    std::vector<T> data(output[0].get_frame_size());
    std::cout << "-I- Started read thread" << std::endl;
    
    auto last_time = std::chrono::high_resolution_clock::now();
    int fps_counter = 0;
    double fps = 0.0;

    while(!should_stop) {
        auto status = output[0].read(MemoryView(data.data(), data.size()));
        if (HAILO_SUCCESS != status){
            should_stop = true;
            return status;
        }

        auto postprocessed_output = scdepth_post_process<T>(data, height, width);

        fps_counter++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();

        if (elapsed_time > 1000) {
            fps = fps_counter * 1000.0 / elapsed_time;
            fps_counter = 0;
            last_time = current_time;
        }

        std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 4);
        cv::putText(postprocessed_output, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Depth Estimation", postprocessed_output);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC
            should_stop = true;
        }
    }

    std::cout << "-I- Finished read thread" << std::endl;
    return HAILO_SUCCESS;
}

void print_net_banner(std::pair< std::vector<InputVStream>, std::vector<OutputVStream> > &vstreams) {
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    std::cout << "-I- Dir  Name                                                          " << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.first){
        std::cout << "-I- IN:  " << value.get_info().name << std::endl;
    }
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.second){
    std::cout << "-I- OUT: " << value.get_info().name << std::endl;
    }
    std::cout << "-I---------------------------------------------------------------------\n" << std::endl;
}

template <typename IN_T, typename OUT_T> hailo_status infer(std::vector<InputVStream> &inputs, std::vector<OutputVStream> &outputs,
                                                            std::string video_path) {
    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;
    
    cv::VideoCapture capture;
    if (video_path.empty()) {
        capture.open(0); // Open default camera
        if (!capture.isOpened()) {
            std::cerr << "-E- Error opening camera" << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }
        std::cout << "-I- Using default camera as input" << std::endl;
    } else {
        capture.open(video_path);
        if (!capture.isOpened()){
            std::cerr << "-E- Error when reading video file: " << video_path << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }
        std::cout << "-I- Using video file as input: " << video_path << std::endl;
    }

    std::atomic<bool> should_stop(false);

    int input_height = inputs.front().get_info().shape.height;
    int input_width = inputs.front().get_info().shape.width;
    int input_channels = inputs.front().get_info().shape.features;
    std::thread input_thread([&]() {
        input_status = write_all<IN_T>(inputs, std::ref(capture), input_height, input_width, input_channels, std::ref(should_stop));
    });

    int output_height = outputs.front().get_info().shape.height;
    int output_width = outputs.front().get_info().shape.width;
    std::thread output_thread([&]() {
        output_status = read_all<OUT_T>(outputs, output_height, output_width, std::ref(should_stop));
    });


    input_thread.join();
    output_thread.join();

    capture.release();
    cv::destroyAllWindows();

    if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
        if (HAILO_SUCCESS != input_status) std::cerr << "-E- Write thread failed with status " << input_status << std::endl;
        if (HAILO_SUCCESS != output_status) std::cerr << "-E- Read thread failed with status " << output_status << std::endl;
        return HAILO_INTERNAL_FAILURE;
    }

    std::cout << "\n-I- Inference finished successfully\n" << std::endl;
    return HAILO_SUCCESS;
}


int main(int argc, char** argv) {
    std::string hef_file   = getCmdOption(argc, argv, "--net", "-n");
    std::string video_path = getCmdOption(argc, argv, "--input", "-i");
    auto all_devices       = Device::scan_pcie();

    if (hef_file.empty()) {
        std::cerr << "-E- No HEF file provided. Use --net or -n to specify the HEF file path." << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }
    
    if (video_path.empty()) {
        std::cout << "-I- No input video path provided, will try to use camera." << std::endl;
    } else {
        std::cout << "-I- video path: " << video_path << std::endl;
    }
    std::cout << "-I- hef: " << hef_file << "\n" << std::endl;

    auto device = Device::create_pcie(all_devices.value()[0]);
    if (!device) {
        std::cerr << "-E- Failed create_pcie " << device.status() << std::endl;
        return device.status();
    }

    auto network_group = configure_network_group(*device.value(), hef_file);
    if (!network_group) {
        std::cerr << "-E- Failed to configure network group " << hef_file << std::endl;
        return network_group.status();
    }

    auto input_vstream_params = network_group.value()->make_input_vstream_params(true, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_vstream_params){
        std::cerr << "-E- Failed make_input_vstream_params " << input_vstream_params.status() << std::endl;
        return input_vstream_params.status();
    }

    auto output_vstream_params = network_group.value()->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_vstream_params){
        std::cerr << "-E- Failed make_output_vstream_params " << output_vstream_params.status() << std::endl;
        return output_vstream_params.status();
    }
    auto input_vstreams  = VStreamsBuilder::create_input_vstreams(*network_group.value(), input_vstream_params.value());
    if (!input_vstreams){
        std::cerr << "-E- Failed create_input_vstreams " << output_vstream_params.status() << std::endl;
        return input_vstreams.status();
    }
    auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group.value(), output_vstream_params.value());
    if (!input_vstreams or !output_vstreams) {
        std::cerr << "-E- Failed creating input: " << input_vstreams.status() << " output status:" << output_vstreams.status() << std::endl;
        return input_vstreams.status();
    }

    
    auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());

    print_net_banner(vstreams);

    auto activated_network_group = network_group.value()->activate();
    if (!activated_network_group) {
        std::cerr << "-E- Failed activated network group " << activated_network_group.status();
        return activated_network_group.status();
    }
    
    auto status  = infer<uint8_t, float32_t>(vstreams.first, vstreams.second, video_path);
    if (HAILO_SUCCESS != status) {
        std::cerr << "-E- Inference failed "  << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}
