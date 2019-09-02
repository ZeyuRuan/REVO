#pragma once
#include "../utils/Logging.h"
#include <Eigen/Eigen>
class PointCloudBuffered
{
public:
    PointCloudBuffered(const Eigen::MatrixXf& pcl, bool hasColor = false)
             : nPoints(pcl.cols()),
               mHasColor(hasColor),
               mSize(hasColor ? 8 : 4)
    {
        glGenBuffers(1, &vbo);
        I3D_LOG(i3d::info) << " Binding: " << (nPoints-1)*mSize*sizeof(float);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER, (nPoints-1)*mSize*sizeof(float), (float*)pcl.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        I3D_LOG(i3d::info) << "After Binding: " << (nPoints-1)*mSize*sizeof(float);
    }
    ~PointCloudBuffered()
    {
        glDeleteBuffers(1, &vbo);
    }
    void drawPoints(const Eigen::Matrix4f& T_w_c)
    {
        GLfloat f = 2.0f;
        glPointSize(f);
        mHasColor = true;
        //the idea is to use the pose

        Eigen::Matrix4f Twc = T_w_c.cast<float>();
       
        glPushMatrix();

        glMultMatrixf((float*)Twc.data());

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableClientState(GL_VERTEX_ARRAY);
        //XYZ1
        glVertexPointer(3, GL_FLOAT, sizeof(float)*8, 0);
        if (mHasColor) glEnableClientState(GL_COLOR_ARRAY);
        //XYZ1,RGB1
        if (mHasColor) glColorPointer(3, GL_FLOAT, sizeof(float)*8, (void *)(sizeof(float) * 4));//);
        glDrawArrays(GL_POINTS,0, nPoints-1);
        if (mHasColor) glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glPopMatrix();
    }
    const int nPoints;
private:

    GLuint vbo;
    bool mHasColor;
    const int mSize; //4 or 8 depending on XYZ1 or XYZ1RGB1

};
