#pragma once
/*
主要定义使用到的数据结构
*/
#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"
#include "bytetrack/include/STrack.h"
#include "../yolo.hpp"
#include "../utils.h"
//#include "app_yolo/yolo.hpp"

struct FrameInfo2Threads
{
    ///
    FrameInfo2Threads()
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }
    ///
    FrameInfo2Threads(size_t batchSize)
        : m_batchSize(batchSize)
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void SetBatchSize(size_t batchSize)
    {
        m_batchSize = batchSize;
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void CleanRegions()
    {
        if (m_regions.size() != m_batchSize)
            m_regions.resize(m_batchSize);
        for (auto& regions : m_regions)
        {
            regions.clear();
        }
    }

    ///
    void CleanTracks()
    {
        if (m_tracks.size() != m_batchSize)
            m_tracks.resize(m_batchSize);
        for (auto& tracks : m_tracks)
        {
            tracks.clear();
        }
    }

    std::vector<std::shared_ptr<Yolo::Image>> m_frames;
    //std::shared_ptr<Yolo::Image> m_uframe; // nvidia 解码的数据
    std::vector<ObjectBox::BoxArray> m_regions;
    std::vector<std::vector<STrack>> m_tracks;
    std::vector<int> m_frameInds;

    size_t m_batchSize = 1;

    int64 m_dt = 0; // 统计耗时

    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::atomic<bool> m_captured{ false };
};
