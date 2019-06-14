#include "CMeshModelInstance.h"
#include <Eigen/Dense>
#include <igl/per_vertex_normals.h>
#include <assert.h>
#include <totalmodel.h>
#include <ceres/rotation.h>
#include <chrono>

// Function equivalent and improved from igl::per_vertex_normals
// [270, 452] ms
template <typename T>
inline T getNormTriplet(const T* const ptr)
{
    return std::sqrt(ptr[0]*ptr[0] + ptr[1]*ptr[1] + ptr[2]*ptr[2]);
}
template <typename T>
inline void normalizeTriplet(T* ptr, const T norm)
{
    ptr[0] /= norm;
    ptr[1] /= norm;
    ptr[2] /= norm;
}
template <typename T>
inline void normalizeTriplet(T* ptr)
{
    const auto norm = getNormTriplet(ptr);
    normalizeTriplet(ptr, norm);
}
void per_vertex_normals(
  const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& V,
  const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>& F,
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& N
)
{
    Eigen::Matrix<double, Eigen::Dynamic,3, Eigen::RowMajor> FN;
    FN.resize(F.rows(),3);
    auto* FN_data = FN.data();
    const auto* const F_data = F.data();
    const auto* const V_data = V.data();
    // loop over faces
    for (int i = 0; i < F.rows();i++)
    {
        const auto baseIndex = 3*i;
        const auto F_data0 = 3*F_data[baseIndex];
        const auto F_data1 = 3*F_data[baseIndex+1];
        const auto F_data2 = 3*F_data[baseIndex+2];
        const Eigen::Matrix<double, 1, 3> v1(
            V_data[F_data1] - V_data[F_data0],
            V_data[F_data1+1] - V_data[F_data0+1],
            V_data[F_data1+2] - V_data[F_data0+2]);
        const Eigen::Matrix<double, 1, 3> v2(
            V_data[F_data2] - V_data[F_data0],
            V_data[F_data2+1] - V_data[F_data0+1],
            V_data[F_data2+2] - V_data[F_data0+2]);
        FN.row(i) = v1.cross(v2);
        auto* fnRowPtr = &FN_data[baseIndex];
        const double norm = getNormTriplet(fnRowPtr);
        if (norm == 0)
        {
            fnRowPtr[0] = 0;
            fnRowPtr[1] = 0;
            fnRowPtr[2] = 0;
        }
        else
            normalizeTriplet(fnRowPtr, norm);
    }

    // Resize for output
    N.resize(V.rows(),3);
    std::fill(N.data(), N.data() + N.rows() * N.cols(), 0.0);

    Eigen::Matrix<double, Eigen::Dynamic, 1> A(F.rows(), 1);
    auto* A_data = A.data();
    const auto Fcols = F.cols();
    const auto Vcols = V.cols();

  // Projected area helper
  const auto & proj_doublearea =
    [&V_data,&F_data, &Vcols, &Fcols](const int x, const int y, const int f)
    ->double
  {
    const auto baseIndex = f*Fcols;
    const auto baseIndex2 = F_data[baseIndex + 2]*Vcols;
    const auto rx = V_data[F_data[baseIndex]*Vcols + x] - V_data[baseIndex2 + x];
    const auto sx = V_data[F_data[baseIndex + 1]*Vcols + x] - V_data[baseIndex2 + x];
    const auto ry = V_data[F_data[baseIndex]*Vcols + y] - V_data[baseIndex2 + y];
    const auto sy = V_data[F_data[baseIndex + 1]*Vcols + y] - V_data[baseIndex2 + y];
    return rx*sy - ry*sx;
  };

  for (auto f = 0;f<F.rows();f++)
  {
    const auto dblAd1 = proj_doublearea(0,1,f);
    const auto dblAd2 = proj_doublearea(1,2,f);
    const auto dblAd3 = proj_doublearea(2,0,f);
    A_data[f] = std::sqrt(dblAd1*dblAd1 + dblAd2*dblAd2 + dblAd3*dblAd3);
  }

    auto* N_data = N.data();
    // loop over faces
    for (int i = 0 ; i < F.rows();i++)
    {
        const auto baseIndex = i*Fcols;
        // throw normal at each corner
        for (int j = 0; j < 3;j++)
        {
            // auto* nRowPtr = &N_data[3*F(i,j)];
            auto* nRowPtr = &N_data[3*F_data[baseIndex + j]];
            const auto* const fnRowPtr = &FN_data[3*i];
            for (int subIndex = 0; subIndex < FN.cols(); subIndex++)
                nRowPtr[subIndex] += A_data[i] * fnRowPtr[subIndex];
            // Vector equilvanet
            // N.row(F(i,j)) += A_data[i] * FN.row(i);
        }
    }

    // take average via normalization
    // loop over faces
    for (int i = 0;i<N.rows();i++)
        normalizeTriplet(&N_data[3*i]);
    // Matrix equivalent
    // N.rowwise().normalize();
}

void CMeshModelInstance::RecomputeNormal(const TotalModel& model)
{
    // Compute Normal
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> V_3(m_vertices.size(), 3);
    auto* V_3_data = V_3.data();

    for (int r = 0; r < V_3.rows(); ++r)
    {
        auto* v3rowPtr = &V_3_data[3*r];
        v3rowPtr[0] = m_vertices[r].x; // V_3(r, 0)
        v3rowPtr[1] = m_vertices[r].y; // V_3(r, 1)
        v3rowPtr[2] = m_vertices[r].z; // V_3(r, 2)
    }
    // Eigen::MatrixXd NV;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> NV;

    if (m_meshType==MESH_TYPE_SMPL)
    {
        std::string errorMessage("Not supporting MESH_TYPE_SMPL currently");
        throw std::runtime_error(errorMessage);
        // igl::per_vertex_normals(V_3, g_smpl.faces_, NV);
    }
    if (m_meshType == MESH_TYPE_TOTAL || m_meshType == MESH_TYPE_ADAM)
    {
        // igl::per_vertex_normals(V_3, model.m_faces, NV);
        per_vertex_normals(V_3, model.m_faces, NV);
        // Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> NVAux;
        // igl::per_vertex_normals(V_3, model.m_faces, NVAux);
        // std::cout << (NV - NVAux).norm() << std::endl;
        // assert((NV - NVAux).norm() < 1e-6);
    }
    m_normals.resize(NV.rows());
    auto* NV_data = NV.data();
    for (int r = 0; r < NV.rows(); ++r)
    {
        const auto* const nvRow = &NV_data[3*r];
        m_normals[r] = cv::Point3f(nvRow[0], nvRow[1], nvRow[2]); // cv::Point3f(NV(r, 0), NV(r, 1), NV(r, 2))
    }
}

void CMeshModelInstance::clearMesh()
{
    m_face_vertexIndices.clear();
    m_vertices.clear();
    m_colors.clear();
    m_normals.clear();
    m_uvs.clear();
    m_joints.clear();
    m_joints_regress.clear();
    m_alpha.clear();
}

bool compareTupleDepth(const std::tuple<double, double, cv::Point3i>& a, const std::tuple<double, double, cv::Point3i>& b)
{
    return std::get<1>(a) > std::get<1>(b);   // from far to near
}

bool compareTupleAlpha(const std::tuple<double, double, cv::Point3i>& a, const std::tuple<double, double, cv::Point3i>& b)
{
    return std::get<0>(a) > std::get<0>(b);   // opaque first, then transparent
}

void CMeshModelInstance::sortFaceDepth(const cv::Point3d angleaxis)
{
    // const auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> depth_vertex(m_vertices.size());
    if (angleaxis == cv::Point3d(0., 0., 0.))   // no rotation
    {
        for (auto i = 0u; i < m_vertices.size(); i++)
            depth_vertex[i] = m_vertices[i].z;
    }
    else
    {
        const double angle_axis[3] = {angleaxis.x, angleaxis.y, angleaxis.z};
        for (auto i = 0u; i < m_vertices.size(); i++)
        {
            const double pt[3] = {m_vertices[i].x, m_vertices[i].y, m_vertices[i].z};
            double result[3];
            ceres::AngleAxisRotatePoint(angle_axis, pt, result);
            depth_vertex[i] = result[2];
        }
    }

    assert(m_face_vertexIndices.size() % 3 == 0);
    std::vector<std::tuple<double, double, cv::Point3i>> vecSort;
    const uint num_face = m_face_vertexIndices.size() / 3;
    vecSort.reserve(num_face);
    for (auto i = 0u; i < num_face; i++)
    {
        const uint I1 = m_face_vertexIndices[3 * i];
        const uint I2 = m_face_vertexIndices[3 * i + 1];
        const uint I3 = m_face_vertexIndices[3 * i + 2];
        const double depth = (depth_vertex[I1] + depth_vertex[I2] + depth_vertex[I3]) / 3;
        const double alpha = (m_alpha[I1] + m_alpha[I2] + m_alpha[I3]) / 3;  // opaqueness
        vecSort.emplace_back(std::make_tuple(alpha, depth, cv::Point3i(I1, I2, I3)));
    }
    std::sort(vecSort.begin(), vecSort.end(), compareTupleAlpha);  // first sort according to opacity, put opaque face first
    auto it = vecSort.begin();
    while(it != vecSort.end())
    {
        // find the first element that is not completely opaque
        if (std::get<0>(*it) != 1.0)
            break;
        it++;
    }
    std::sort(it, vecSort.end(), compareTupleDepth);  // now sort according to depth, far first
    for (auto i = 0u; i < num_face; i++)
    {
        auto& pt = std::get<2>(vecSort[i]);
        m_face_vertexIndices[3 * i + 0] = pt.x;
        m_face_vertexIndices[3 * i + 1] = pt.y;
        m_face_vertexIndices[3 * i + 2] = pt.z;
    }
    // const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    // std::cout << "Sort depth duration " << duration * 1e-6 << " ms" << std::endl;
}