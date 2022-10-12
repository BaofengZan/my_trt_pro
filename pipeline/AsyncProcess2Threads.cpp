#include "AsyncProcess2Threads.h"
#include <memory>
#include <future>
#include <chrono>
#include "../application/yolo/yolo.hpp"
#include "../tensorrt/trt_infer.hpp"
#include "../ffhdd/ffmpeg_demuxer.hpp"
#include "../ffhdd/cuvid_decoder.hpp"
#include "../ffhdd/nalu.hpp"

AsyncProcess2Threads::AsyncProcess2Threads(const std::shared_ptr<Yolo::Infer>& yolo, std::string & inFile, int id):m_id(id)
{
    yolo_ = yolo;
    m_inFile = inFile;
    m_outFile = inFile.replace(m_inFile.find("."), 4, "_" + std::to_string(m_id) + ".mp4");

    m_colors.emplace_back(255, 0, 0);
    m_colors.emplace_back(0, 255, 0);
    m_colors.emplace_back(0, 0, 255);
    m_colors.emplace_back(255, 255, 0);
    m_colors.emplace_back(0, 255, 255);
    m_colors.emplace_back(255, 0, 255);
    m_colors.emplace_back(255, 127, 255);
    m_colors.emplace_back(127, 0, 255);
    m_colors.emplace_back(127, 0, 127);
}

void AsyncProcess2Threads::AsyncProcess()
{
    std::atomic<bool> stopCapture(false);

    std::thread thCapDet(&AsyncProcess2Threads::CaptureAndDetect, this, std::ref(stopCapture));

    cv::VideoWriter writer;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;
    int64 startLoopTime = cv::getTickCount();
    size_t processCounter = 0;
    for (; !stopCapture.load(); )
    {
        FrameInfo2Threads& frameInfo = m_frameInfo[processCounter % 2];
        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            if (!frameInfo.m_cond.wait_for(lock, std::chrono::milliseconds(m_captureTimeOut), [&frameInfo] { return frameInfo.m_captured.load(); }))
            {
                std::cout << "--- Wait frame timeout!" << std::endl;
                break;
            }
        }
        if (!m_isTrackerInitialized)
        {
            tracker = std::make_shared<BYTETracker>(m_fps, 10);
            m_isTrackerInitialized = true;
            if (!m_isTrackerInitialized)
            {
                std::cerr << "--- AsyncProcess: Tracker initialize error!!!" << std::endl;
                frameInfo.m_cond.notify_one();
                break;
            }
        }

        int64 t1 = cv::getTickCount();

        Tracking(frameInfo);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1 + frameInfo.m_dt;
        int currTime = cvRound(1000 * (t2 - t1 + frameInfo.m_dt) / freq);


        int key = 0;
        for (size_t i = 0; i < m_batchSize; ++i)
        {
            //DrawData(frameInfo.m_frames[i].GetMatBGR(), frameInfo.m_tracks[i], frameInfo.m_frameInds[i], currTime);
            //auto& output_stracks = m_frameInfo->m_tracks[i];
            auto& output_stracks = frameInfo.m_tracks[i];
            if (output_stracks.empty())
            {
                continue;
            }
            cv::Mat img = frameInfo.m_frames[i]->cvmat.clone();
            //for (auto& obj : frameInfo->m_regions) {
            //    cv::rectangle(img, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(0., 0, 255), 2);
            //}

            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                //bool vertical = tlwh[2] / tlwh[3] > 1.6;
                bool vertical = false;
                if (tlwh[2] * tlwh[3] > 0 && !vertical)
                {
                    cv::Scalar s = tracker->get_color(output_stracks[i].track_id);
                    putText(img, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

                    cv::putText(img, "fps: " + std::to_string(1000.0f / (1.0f*frameInfo.m_dt)), cv::Point(20, 20),
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

                    cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
                }
            }
            //WriteFrame(writer, img);
            cv::imshow(std::to_string(m_id), img);
            cv::waitKey(1);
        }

        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            assert(frameInfo.m_captured.load());
            frameInfo.m_captured = false;
        }
        frameInfo.m_cond.notify_one();

        if (key == 27)
            break;

        ++processCounter;
    }
    stopCapture = true;

    if (thCapDet.joinable())
        thCapDet.join();

    int64 stopLoopTime = cv::getTickCount();

    std::cout << "--- algorithms time = " << (allTime / freq) << ", work time = " << ((stopLoopTime - startLoopTime) / freq) << std::endl;

}

std::vector<Object> AsyncProcess2Threads::BoxArray2TrackBox(const ObjectBox::BoxArray & det_box)
{
    std::vector<Object> objs;
    for (const auto& item : det_box)
    {
        Object obj;
        obj.rect = cv::Rect(cv::Point(item.left, item.top), cv::Point(item.right, item.bottom));
        obj.label = item.class_label;
        obj.prob = item.confidence;
        objs.emplace_back(obj);
    }

    return objs;
}

void AsyncProcess2Threads::CaptureAndDetect(std::atomic<bool>& stopCapture)
{
    auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(m_inFile);
    if (demuxer == nullptr) {
        return;
    }

    int decoder_device_id = 0;
    auto decoder = FFHDDecoder::create_cuvid_decoder(
        true, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, decoder_device_id
    );

    if (decoder == nullptr) {
        return;
    }

    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    int64_t pts = 0;

    /* 这个是头，具有一些信息，但是没有图像数据 */
    demuxer->get_extra_data(&packet_data, &packet_size);
    decoder->decode(packet_data, packet_size);


    int framesCounter = 0;

    const auto localEndFrame = this->m_endFrame;
    auto localIsDetectorInitialized = this->m_isDetectorInitialized;
    auto localTrackingTimeOut = this->m_trackingTimeOut;
    size_t processCounter = 0;
    for (; !stopCapture.load()&& (packet_size > 0);)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        FrameInfo2Threads& frameInfo = this->m_frameInfo[processCounter % 2];
        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            if (!frameInfo.m_cond.wait_for(lock, std::chrono::milliseconds(localTrackingTimeOut), [&frameInfo] { return !frameInfo.m_captured.load(); }))
            {
                std::cout << "+++ Wait tracking timeout!" << std::endl;
                frameInfo.m_cond.notify_one();
                break;
            }
        }
        if (frameInfo.m_frames.size() < frameInfo.m_batchSize)
        {
            frameInfo.m_frames.resize(frameInfo.m_batchSize);
            frameInfo.m_frameInds.resize(frameInfo.m_batchSize);
        }


        size_t i = 0;
        for (; i < frameInfo.m_batchSize; ++i)
        {
            demuxer->demux(&packet_data, &packet_size, &pts);
            int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
            for (int j = 0; j < ndecoded_frame; ++j) {

                unsigned int frame_index = 0;
                frameInfo.m_frames[i] = std::make_shared<Yolo::Image>(decoder->get_frame(&pts, &frame_index),
                    decoder->get_width(), decoder->get_height(),
                    decoder_device_id);
                frameInfo.m_frameInds[i] = framesCounter;
                ++framesCounter;

                if (localEndFrame && framesCounter > localEndFrame)
                {
                    std::cout << "+++ Process: riched last " << localEndFrame << " frame" << std::endl;
                    break;
                }
            }
           
        }
        if (i < frameInfo.m_batchSize)
            break;

        //if (!localIsDetectorInitialized)
        //{
        //    thisPtr->m_isDetectorInitialized = this->InitDetector(frameInfo.m_frames[0].GetUMatBGR());
        //    localIsDetectorInitialized = thisPtr->m_isDetectorInitialized;
        //    if (!thisPtr->m_isDetectorInitialized)
        //    {
        //        std::cerr << "+++ CaptureAndDetect: Detector initialize error!!!" << std::endl;
        //        frameInfo.m_cond.notify_one();
        //        break;
        //    }
        //}

        //CleanRegions
        frameInfo.CleanRegions();
        for (int i = 0; i < frameInfo.m_frames.size(); ++i)
        {
            if (frameInfo.m_frames[i] == nullptr)
            {
                continue;
            }
            std::shared_future<ObjectBox::BoxArray> predbox = yolo_->commit(*frameInfo.m_frames[i]);
            //cv::imshow("ddbg", frameInfo.m_frames[i]->cvmat);
            //cv::waitKey(0);
            const ObjectBox::BoxArray& regions = predbox.get();
            frameInfo.m_regions[i]= regions;
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
        printf("第%d路, 解码-检测-跟踪耗时 %f ms \n", m_id, duration_ms.count());
        //int64 t2 = cv::getTickCount();
        frameInfo.m_dt = duration_ms.count();

        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            assert(!frameInfo.m_captured.load());
            frameInfo.m_captured = true;
        }
        frameInfo.m_cond.notify_one();

        ++processCounter;
    }
    stopCapture = true;
}

void AsyncProcess2Threads::Tracking(FrameInfo2Threads & frame)
{
    frame.CleanTracks();
    for (size_t i = 0; i < frame.m_frames.size(); ++i)
    {
        if (frame.m_frames[i] == nullptr)
        {
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<STrack> output_stracks = tracker->update(BoxArray2TrackBox(frame.m_regions[i]));
        frame.m_tracks[i].insert(frame.m_tracks[i].end(), output_stracks.begin(), output_stracks.end());
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
        frame.m_dt += duration_ms.count();
    }

}

bool AsyncProcess2Threads::WriteFrame(cv::VideoWriter & writer, const cv::Mat & frame)
{
    if (!m_outFile.empty())
    {
        if (!writer.isOpened()) {
            writer.open(m_outFile, cv::CAP_OPENCV_MJPEG, m_fps, frame.size(), true);
        }
        if (writer.isOpened())
        {
            writer << frame;
            return true;
        }
    }
    return false;
}
