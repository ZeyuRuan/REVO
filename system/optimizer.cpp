/*
 * This optimizer is a modified version of LSD-SLAM's SE3 Tracker http://vision.in.tum.de/lsdslam
 * "LSD-SLAM: Large-scale direct monocular SLAM, Engel et al., ECCV 2014"
*/

#include "optimizer.h"
#include <Eigen/Core>
#include "../utils/Logging.h"
#include "../utils/timer.h"
#include <memory>
#include "sophus/se3.hpp"
#include <opencv2/highgui.hpp>

void reprojectRefEdgesToCurrentFrameop(const cv::Mat& rgbCurr,const Eigen::MatrixXf& _3d, const Eigen::Matrix3f& K,
                                        const Eigen::Matrix3f& R, const Eigen::Vector3f &t, const cv::Mat& edgesCurr,
                                        const std::string title);
Optimizer::Optimizer(const OptimizerSettings &settings): mSettings(settings)
{

    auto startOpt = Timer::getTime();

    const int w = mSettings.maxImgSize.width;
    const int h = mSettings.maxImgSize.height;
    size_t memSize = w * h * sizeof(float) * 1;
    buf_warped_residual = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_dx = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_dy = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_x = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_y = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_z = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_weight_p = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    auto endOpt = Timer::getTime();
    I3D_LOG(i3d::info) << "END Constructing Optimizer" << Timer::getTimeDiffMiS(startOpt,endOpt);
}
Optimizer::~Optimizer()
{
    Eigen::internal::aligned_free(static_cast<void*>(buf_warped_residual));
    Eigen::internal::aligned_free(static_cast<void*>(buf_warped_dx));
    Eigen::internal::aligned_free(static_cast<void*>(buf_warped_dy));
    Eigen::internal::aligned_free(static_cast<void*>(buf_warped_x));
    Eigen::internal::aligned_free(static_cast<void*>(buf_warped_y));
    Eigen::internal::aligned_free(static_cast<void*>(buf_warped_z));
    Eigen::internal::aligned_free(static_cast<void*>(buf_weight_p));
}

//think about filling up the buffers
float Optimizer::calcErrorAndBuffers(const std::shared_ptr<ImgPyramidRGBD> & refFrame, const std::shared_ptr<ImgPyramidRGBD> & currFrame,
                                     const Eigen::Matrix3f& R, const Eigen::Vector3f& T,ResidualInfo& resInfo,
                                     const uint lvl, bool FILL_BUFFERS)
{
    LOG_THRESHOLD(i3d::nothing);
    resInfo.clearAll();
    const Camera cam = refFrame->cameraPyr->at(lvl);
    const int w = cam.width, h = cam.height;
    Eigen::Matrix<float,3,4> transform = Eigen::Matrix<float,3,4>::Zero(3,4);
    transform.block<3,3>(0,0) = R;
    transform.block<3,1>(0,3) = T;
    const Eigen::Matrix4Xf& currPcl = currFrame->return3DEdges(lvl);
    const Eigen::Vector4f* distGradients = refFrame->returnOptimizationStructure(lvl);

    I3D_LOG(i3d::info) << "Cam: " << std::fixed << cam.fx << " " << cam.fy << " " << cam.cx << " " << cam.cy;
    for (int c = 0; c < currPcl.cols();++c)
    {
        const Eigen::Vector4f refPoint = currPcl.col(c); //first three are the vector
        //IDEA: Use far points only for rotation!
        // (u_new, v_new): new point reprojected to frame
        const Eigen::Vector3f Wxp = R * refPoint.head<3>() + T;
        const float u_new = Wxp[0]/Wxp[2]*cam.fx + cam.cx;
        const float v_new = Wxp[1]/Wxp[2]*cam.fy + cam.cy;

        // step 1a: coordinates have to be in image:
        // (inverse test to exclude NANs)
        //in image?
        if(!(u_new > 1 && v_new > 1 && u_new < w-2 && v_new < h-2))
        {
            resInfo.badPtsEdges++;
            I3D_LOG(i3d::info) << std::fixed << resInfo.badPtsEdges << " out of bounds: " << u_new << " " << v_new << "Z: " << refPoint[2] << " " << Wxp.transpose() ;
            continue;
        }
        const Eigen::Vector3f resInterp = getInterpolatedElement43(distGradients, u_new, v_new, w);
        const float residual = resInterp[2];
        if ((residual > mSettings.edgeDistanceLvl[lvl]) && mSettings.USE_EDGE_FILTER)
        {
            resInfo.badPtsEdges++;
            continue;
        }
        const float w_r  = getWeightOfEvoR(residual);
        if (FILL_BUFFERS)
        {
            const int eIdx = resInfo.goodPtsEdges;
            *(buf_warped_x+eIdx) = Wxp[0];
            *(buf_warped_y+eIdx) = Wxp[1];
            *(buf_warped_z+eIdx) = Wxp[2];
            *(buf_warped_dx+eIdx) = cam.fx * resInterp[0];
            *(buf_warped_dy+eIdx) = cam.fy * resInterp[1];
            *(buf_warped_residual+eIdx) = residual;
            *(buf_weight_p+eIdx) = w_r;
        }
        if (resInfo.goodPtsEdges < 30)
        {
            I3D_LOG(i3d::info) << std::fixed << resInfo.goodPtsEdges << ": " <<"(" << u_new <<", " << v_new << "):" << Wxp.transpose()<< "Z: " << refPoint[2];
        }
        const float res_2 = residual*residual;
        //I3D_LOG(i3d::info) << "residual: "<<residual << " resInfo: " << resInfo.sumErrorWeighted << " " << w_r;
        resInfo.sumErrorWeighted += (w_r*res_2);
        resInfo.sumErrorUnweighted += res_2;
        resInfo.goodPtsEdges++;
    }
    LOG_THRESHOLD(i3d::debug);
    I3D_LOG(i3d::info) <<std::fixed<< "goodCount: " << (resInfo.goodPtsEdges) << "; goodEdges: " << resInfo.goodPtsEdges <<
                        "; badEdges: " << resInfo.badPtsEdges << "; sumErrorUnweighted: " << resInfo.sumErrorUnweighted << "; sumErrorWeighted: " << resInfo.sumErrorWeighted ;

    return resInfo.sumErrorWeighted / (resInfo.goodPtsEdges);
}
void Optimizer::calculateWarpUpdate(LGS6 &ls, int goodPoints)
{
    I3D_LOG(i3d::info) << "Computing warp update for "<<goodPoints<<" residuals";
    ls.initialize(goodPoints);
    //int nEdges = 0;
    for(int i=0;i<goodPoints;i++)
    {
        float px = *(buf_warped_x+i);
        float py = *(buf_warped_y+i);
        float pz = *(buf_warped_z+i);
        float r =  *(buf_warped_residual+i);
        float gx = *(buf_warped_dx+i);
        float gy = *(buf_warped_dy+i);


        // step 3 + step 5 compute 6d error vector
        //isDepth = false;

        Vector6 v;

        float z = 1.0f / pz;
        float z_sqr = 1.0f / (pz*pz);

        //the derivation is defined in Kerl's Master's Thesis, p. 34
        //https://vision.in.tum.de/_media/spezial/bib/kerl2012msc.pdf
        //fx and fy were premultiplied onto the gradient!
        v[0] = z*gx + 0;
        v[1] = 0 +         z*gy;
        v[2] = (-px * z_sqr) * gx +
              (-py * z_sqr) * gy;

        v[3] = (-px * py * z_sqr) * gx +
              (-(1.0 + py * py * z_sqr)) * gy;
        v[4] = (1.0 + px * px * z_sqr) * gx +
              (px * py * z_sqr) * gy;
        v[5] = (-py * z) * gx +
              (px * z) * gy;
        // step 6: integrate into A and b:
        ls.update(v, r, *(buf_weight_p+i));
    }
    // solve ls
    ls.finish();
}

// compute optimal R,t, return the smallest error
float Optimizer::trackFrames(const std::shared_ptr<ImgPyramidRGBD> &refFrame, const std::shared_ptr<ImgPyramidRGBD> &currFrame,
                             Eigen::Matrix3f &R, Eigen::Vector3f &T, int lvl, ResidualInfo& resInfo)
{
    I3D_LOG(i3d::info) << "Track Frames!\n  R = \n" << R << " \n  t = " << T;
    // ============ track frame ============
    //Sophus::SE3d referenceToFrame(R.cast<double>(),T.cast<double>());
    Sophus::SE3f referenceToFrame(R,T);
    lsd_slam::LGS6 ls;
    float lastErr = calcErrorAndBuffers(refFrame,currFrame,R,T,resInfo,lvl);
    //exit(0);
    float last_residual = lastErr;
    float LM_lambda = mSettings.lambdaInitial[lvl];
    I3D_LOG(i3d::info) << "LM_lambda: " << LM_lambda;
    //int iterationNumber = 0;
    ///NOTE: We might need MAX_LVL-lvl or something like that
    for(int iteration=0; iteration < mSettings.maxItsPerLvl[lvl]; iteration++)
    {
        calculateWarpUpdate(ls,resInfo.goodPtsEdges);
        int incTry=0;
        I3D_LOG(i3d::info) << "calculateWarpUpdate: " << resInfo.goodPtsEdges;
        while(true)
        {
            // solve LS system with current lambda
            Vector6 b = -ls.b;
            Matrix6x6 A = ls.A;
            //I3D_LOG(i3d::info)  << "Before lambda A: " <<A << "b: "<<b.transpose() << LM_lambda;
            for(int i=0;i<6;i++) A(i,i) *= 1+LM_lambda;
            Vector6 inc = A.ldlt().solve(b);
            incTry++;
            // apply increment. pretty sure this way round is correct, but hard to test.
            //Sophus::SE3d new_referenceToFrame = Sophus::SE3d::exp(inc.cast<double>()) * referenceToFrame;
            Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp(inc) * referenceToFrame;
            // re-evaluate residual
            float error = calcErrorAndBuffers(refFrame,currFrame,new_referenceToFrame.rotationMatrix()/*.cast<float>()*/,
                                              new_referenceToFrame.translation()/*.cast<float>()*/,resInfo,lvl);
            I3D_LOG(i3d::detail) << "After calcResidualAndBuffersEdges: " << error;
            I3D_LOG(i3d::info) <<"goodPts: " << resInfo.goodPtsEdges << "; bad: " << resInfo.badPtsEdges << "; total: " << resInfo.goodPtsEdges+resInfo.badPtsEdges<<"; error = "<< error <<" = " << resInfo.sumErrorWeighted << "/" << resInfo.goodPtsEdges;
            // accept inc?
            if(error < lastErr)
            {
                // accept inc
                referenceToFrame = new_referenceToFrame;
                // converged?
                if(error / lastErr > mSettings.convergenceEps[lvl])
                {
                    I3D_LOG(i3d::debug) << "(" << lvl <<", "<< iteration<< "," << error / lastErr <<" ): FINISHED pyramid level (last residual reduction too small).";
                    iteration = mSettings.maxItsPerLvl[lvl];
                    //final_uncertainty = uncertainty;
                }
                last_residual = lastErr = error;
                if(LM_lambda <= 0.2f)
                    LM_lambda = 0.0f;
                else
                    LM_lambda *= mSettings.lambdaSuccessFac;
                //I3D_LOG(i3d::info) << "Increment accepted: " << inc.transpose();
                break;
            }
            else
            {
                if(!(inc.dot(inc) > mSettings.stepSizeMin[lvl]))
                {
                    I3D_LOG(i3d::debug) << "(" << lvl <<", "<< iteration<<"): FINISHED pyramid level (stepsize too small).";
                    iteration = mSettings.maxItsPerLvl[lvl];
                    break;
                }
                if(LM_lambda == 0.0f)
                    LM_lambda = 0.2f;
                else
                    LM_lambda *= std::pow(mSettings.lambdaFailFac, incTry);
            }
        }
        I3D_LOG(i3d::info) << "incTry: " << incTry;
    }
    R = referenceToFrame.rotationMatrix();//.cast<float>();
    T = referenceToFrame.translation();//.cast<float>();
    return last_residual;
}


