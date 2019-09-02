#pragma once
#include <string>
#include "optimizer.h"
#include "tracker.h"

#ifdef WITH_PANGOLIN_VIEWER
    #include "../gui/Viewer.h"
#endif
#include "../datastructures/imgpyramidrgbd.h"
#include "../io/iowrapperRGBD.h"
#include <memory.h>
#include "../utils/timer.h"

class REVOConfig
{
public:
    REVOConfig(const std::string& settingsFile,const std::string& dataSettings, int nRuns = 0):
        settingsPyr(dataSettings),settingsIO(dataSettings,nRuns),settingsTracker(settingsFile)
    {
        //Handle output through IOWrapper
        //Initialize the thing using OpenCV

        I3D_LOG(i3d::info) << "MainFolder: " <<settingsIO.MainFolder+"/"+settingsIO.subDataset+"/" << nRuns;
	
	// store settingsfile in dataF
        cv::FileStorage dataF(settingsFile,cv::FileStorage::READ);
        if (!dataF.isOpened())
        {
            I3D_LOG(i3d::error) << "Couldn't open settings file at location: " << settingsFile;
            exit(0);
        }
        else
            I3D_LOG(i3d::error) << "Reading from settings file at location: " << settingsFile;

        //image pyramid settings (camera matrix, resolutions,...)
        cv::read(dataF["GT_POSEREADER_ACTIVE"],GT_POSEREADER_ACTIVE,false);
        cv::read(dataF["INIT_FROM_LAST_POSE"],INIT_FROM_LAST_POSE,true);
        cv::read(dataF["DO_OUTPUT_POSES"],DO_OUTPUT_POSES,false);
        cv::read(dataF["DO_RECORD_IMAGES"],DO_RECORD_IMAGES,false);
        cv::read(dataF["DO_USE_PANGOLIN_VIEWER"],DO_USE_PANGOLIN_VIEWER,true);
        cv::read(dataF["DO_SHOW_DEBUG_IMAGE"],DO_SHOW_DEBUG_IMAGE,false);
        I3D_LOG(i3d::info) << "DO_SHOW_DEBUG_IMAGE = " << DO_SHOW_DEBUG_IMAGE;
        cv::read(dataF["DO_GENERATE_DENSE_PCL"],DO_GENERATE_DENSE_PCL,false);
        dataF.release();
    }
    //only relevant when keyframes are used
    bool INIT_FROM_LAST_POSE;
    //number of frames that should be skipped (only relevant for pre-computed edges)
    //unsigned int skipFirstFrames;
    bool GT_POSEREADER_ACTIVE;
    std::string dataFolder;
    std::string subDataset;
    std::string outputFolder;
    ImgPyramidSettings settingsPyr;
    //Eigen::Matrix3f K;
    //double distThreshold, angleThreshold;
    bool READ_FROM_ASTRA;
    bool DO_OUTPUT_POSES;
    bool DO_USE_PANGOLIN_VIEWER;
    bool DO_SHOW_DEBUG_IMAGE;
    bool DO_GENERATE_DENSE_PCL;
    IOWrapperSettings settingsIO;
    TrackerSettings settingsTracker;
    bool DO_RECORD_IMAGES;
    //WindowedOptimizationSettings settingsWO;
};

class REVO
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    class Pose
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        Pose(const Eigen::Matrix4f& T_kf_curr, double timestamp, const std::shared_ptr<ImgPyramidRGBD>& _kfFrame):
            T_kf_curr(T_kf_curr),isKeyFrame(false),timestamp(timestamp), kfFrame(_kfFrame)
        {

        }
	// new frame = current frame
	// T_W_N = T_W_F*T_kf_curr : transformation from new frame to world
	// T_N_W = T_W_N.inv
        inline const Eigen::Matrix4f T_N_W() const
        {
            return (kfFrame->getTransKFtoWorld()*T_kf_curr).inverse();
        }
        inline const Eigen::Matrix4f T_W_N() const
        {
            return getCurrToWorld();
        }
	// T_N_kf: key frame to new frame(current frame)
        inline const Eigen::Matrix4f T_N_kf() const
        {
            return this->T_kf_curr.inverse();
        }
        inline const Eigen::Matrix4f& T_kf_N() const
        {
            return this->T_kf_curr;
        }
	// 
        inline const Eigen::Matrix4f getTransKFtoCurr() const
        {
            return this->T_kf_curr.inverse();
        }
        inline const Eigen::Matrix4f& getTransCurrtoKF() const
        {
            return this->T_kf_curr;
        }
        const Eigen::Matrix4f& getPose() const
        {
            return this->T_kf_curr;
        }
        //world pose of current frame is:
        //T_W_curr =  T_W_KF * T_KF_CURR
        inline const Eigen::Matrix4f getCurrToWorld() const
        {
	    // T_w_curr = T_w_kf * T_kf_curr
            return kfFrame->getTransKFtoWorld()*T_kf_curr;
        }
        inline double returnTimestamp() const
        {
            return timestamp;
        }

        //this method is only called when the "previous frame" is taken as keyframe!
        void setKfFrame(const std::shared_ptr<ImgPyramidRGBD>& _kfFrame)
        {
            //Update parent keyframe -> this is typically "itself"
            kfFrame = _kfFrame;
            //no relative transformation to "itself"
            T_kf_curr = Eigen::Matrix4f::Identity();
        }
    private:
        Eigen::Matrix4f T_kf_curr;
        bool isKeyFrame;
        double timestamp;
        std::shared_ptr<ImgPyramidRGBD> kfFrame; //this is the parent frame
    };

    std::vector<Pose,Eigen::aligned_allocator<Eigen::Matrix4f>> mPoseGraph;

    REVO(const std::string &settingsFile, const std::string &dataSettings, int nRuns);
    ~REVO();
    bool start();
    // reject from Point Cloud to Image
    float reprojectPCLToImg(const Eigen::MatrixXf& pcl, const Eigen::Matrix3f &R, const Eigen::Vector3f &T,
                            const cv::Mat& img, const cv::Size2i& size,
                            const Eigen::Matrix3f& K, int& goodCount, int& badCount, const std::__cxx11::string &title) const;
private:
    const REVOConfig mSettings;
    std::shared_ptr<CameraPyr> camPyr;
    std::ofstream mPoseFile;
    std::unique_ptr<TrackerNew> mTracker;
#ifdef WITH_PANGOLIN_VIEWER
    std::shared_ptr<REVOGui::MapDrawer> mpMapDrawer;
    std::shared_ptr<REVOGui::Viewer> mpViewer;
#endif
    std::unique_ptr<std::thread> mThViewer, mThIOWrapper;
    std::shared_ptr<IOWrapperRGBD> mIOWrapper;
    bool flagTrackingLost;
    std::vector<double> trackingTimes;
    std::vector<double> accStructureTimes;
    std::vector<double> dT;
    bool isFinished;

private:

    inline Eigen::Matrix4f transformFromRT(const Eigen::Matrix3f& R, const Eigen::Vector3f& T) const
    {
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3,3>(0,0) = R;
        transform.block<3,1>(0,3) = T;
        return transform;
    }
    TrackerNew::TrackerStatus checkTrackingResult(const Eigen::Matrix4f &currPoseInWorld);
    void writePose(const Eigen::Matrix3f &R, const Eigen::Vector3f &T, const double timeStamp);

    static inline std::string RTToTUMString(const Eigen::Matrix3d& R, const Eigen::Vector3d& T, double timeStamp)
    {
        Eigen::Quaterniond Qf(R);
        std::stringstream tumString;
        tumString << std::fixed << "timeStamp= " << timeStamp << "; T[0]= " << T[0] << "; T[1]= " << T[1] << "; T[2]= " << T[2] << "; x= " <<  Qf.x() << "; y= " << Qf.y() << "; z= " << Qf.z() << "; w= " << Qf.w();
        return tumString.str();
    }
    static inline std::string poseToTUMString(const Eigen::Matrix4d& pose, const double timeStamp)
    {
        Eigen::Matrix3d R = pose.block<3,3>(0,0);
        Eigen::Vector3d T = pose.block<3,1>(0,3);
        return RTToTUMString(R,T,timeStamp);
    }
};
