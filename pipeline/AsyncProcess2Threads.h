#pragma once
//#include "Queue.h"
#include "STrack.h"
#include "BYTETracker.h"
#include "objs_info.hpp"
#include "../utils.h"

class AsyncProcess2Threads
{
public:
    AsyncProcess2Threads(const std::shared_ptr<Yolo::Infer>& yolo, std::string& inFile, int id = 0);
    AsyncProcess2Threads(const AsyncProcess2Threads&) = delete;
    AsyncProcess2Threads(AsyncProcess2Threads&&) = delete;
    AsyncProcess2Threads& operator=(const AsyncProcess2Threads&) = delete;
    AsyncProcess2Threads& operator=(AsyncProcess2Threads&&) = delete;

    virtual ~AsyncProcess2Threads() = default;

    void AsyncProcess();

protected:
    std::vector<Object> BoxArray2TrackBox(const ObjectBox::BoxArray& det_box);
    //std::unique_ptr<BaseDetector> m_detector;
    //std::unique_ptr<BaseTracker> m_tracker;
    std::shared_ptr<Yolo::Infer> yolo_; // 检测器 由外部传入。   来实现多路使用同一个模型
    std::shared_ptr<BYTETracker> tracker;

    float m_fps = 25;
    cv::Size m_frameSize;
    int m_framesCount = 0;

    size_t m_batchSize = 1;

    int m_captureTimeOut = 60000;
    int m_trackingTimeOut = 60000;


    void CaptureAndDetect(std::atomic<bool>& stopCapture);


    void Detection(FrameInfo2Threads& frame);
    void Tracking(FrameInfo2Threads& frame);

    //virtual void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) = 0;
    //virtual void DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter);

    bool m_trackerSettingsLoaded = false;

    std::vector<cv::Scalar> m_colors;

private:
    std::vector<STrack> m_tracks;

    bool m_isTrackerInitialized = false;
    bool m_isDetectorInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    int m_id = 0;
    int m_fourcc = cv::VideoWriter::fourcc('D', 'I', 'V', 'X');
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    
    FrameInfo2Threads m_frameInfo[2];

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};