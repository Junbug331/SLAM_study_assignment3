#include <iostream>
#include <set>
#include <filesystem>
#include <random>

#include <utils.hpp>

namespace fs = std::filesystem;
using namespace std;
using Matrix34d = Eigen::Matrix<double, 3, 4>;

Matrix34d calculateSVD(const Eigen::MatrixXd& input_mat, const std::vector<int> &indexes)
{
    size_t input_rows = indexes.size();
    size_t rows = (input_rows << 1);

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(rows, 12);
    for (int i=0; i<input_rows; i++)
    {
        int r = indexes[i];
        double x = input_mat.coeff(r, 0);
        double y = input_mat.coeff(r, 1);
        double Mx = input_mat.coeff(r, 2);
        double My = input_mat.coeff(r, 3);
        double Mz = input_mat.coeff(r, 4);

        Eigen::MatrixXd t1(1, 12), t2(1, 12);
        t1 << Mx, My, Mz, 1., 0., 0., 0., 0., -x*Mx, -x*My, -x*Mz, -x;
        t2 << 0., 0., 0., 0., Mx, My, Mz, 1., -y*Mx, -y*My, -y*Mz, -y;

        B.block<1, 12>((i<<1), 0) = t1;
        B.block<1, 12>((i<<1)+1, 0) = t2;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(B);
    auto v = svd.matrixV().col(11);
    Matrix34d C = Eigen::MatrixXd::Identity(3, 4);
    C.block<1, 4>(0, 0) = v.block<4, 1>(0, 0);
    C.block<1, 4>(1, 0) = v.block<4, 1>(4, 0);
    C.block<1, 4>(2, 0) = v.block<4, 1>(8, 0);

    return std::move(C);
}

Matrix34d solutionRANSAC(const Eigen::MatrixXd &input_mat, int num_iter, double threshold)
{
    std::random_device rd;
    std::mt19937 eng(rd());
    int last = input_mat.rows();
    std::uniform_int_distribution<int> udist(0, last-1);

    int best_inliers_cnt = 0;
    Matrix34d ans;
    auto[P_I, P_M] =  splitInputMatrix(input_mat);

    for (int i=0; i<num_iter; i++)
    {
        std::set<int> nums;
        while(nums.size() < 11)
            nums.insert(udist(eng));
        std::vector<int> indexes(nums.begin(), nums.end());

        // answer candidate
        Matrix34d C = calculateSVD(input_mat, indexes);

        // get inliers
        int inliers = 0;
        for (int r=0; r<input_mat.rows(); ++r)
        {
            auto tmp = P_M.row(r).transpose();
            auto ref_p = P_I.row(r).transpose();
            Eigen::Vector4d m;
            m << tmp.coeff(0, 0), tmp.coeff(1, 0), tmp.coeff(2, 0), 1.0;
            Eigen::Vector3d p_ = C * m;
            Eigen::Vector2d p;
            double x = p_.coeff(0, 0);
            double y = p_.coeff(1, 0);
            double z = p_.coeff(2, 0);
            p << x/z, y/z;
            if((ref_p - p).norm() < threshold)
                inliers++;
        }

        if (inliers > best_inliers_cnt)
        {
            best_inliers_cnt = inliers;
            ans = C;
        }
    }

    cout << "best inliers count: " << best_inliers_cnt << endl;

    return ans;
}

int main()
{
    fs::path dir(ROOT_DIR);
    fs::path input_file("data.csv");
    fs::path output_file("output.csv");
    fs::path input_path = dir/input_file;
    fs::path output_path = dir/output_file;

    {
        double thresh = 55;
        Eigen::MatrixXd input_mat = readCSVFiletoEigen(input_path, true);
        Eigen::MatrixXd output_mat = readCSVFiletoEigen(output_path);
        auto[P_I, P_M] = splitInputMatrix(input_mat);
        auto C = solutionRANSAC(input_mat, 2500, thresh); // Best threshold : 30~55

        int rows = P_I.rows();

        double mean_diff = 0.0;
        double mean_diff_ans = 0.0;

        for (int r=0; r<rows; ++r)
        {
            auto tmp = P_M.row(r).transpose();
            auto ref_p = P_I.row(r).transpose();
            Eigen::Vector4d m;
            m << tmp.coeff(0, 0), tmp.coeff(1, 0), tmp.coeff(2, 0), 1.0;

            // 
            Eigen::Vector3d p_ = C * m;
            Eigen::Vector2d p;
            double x = p_.coeff(0, 0);
            double y = p_.coeff(1, 0);
            double z = p_.coeff(2, 0);
            p << x/z, y/z;
            mean_diff += (ref_p - p).norm();
            cout << ref_p.transpose() << endl;
            cout << p.transpose()<< endl << endl;

            // GT
            Eigen::Vector3d p_ans = output_mat * m;
            Eigen::Vector2d q;
            x = p_ans.coeff(0, 0);
            y = p_ans.coeff(1, 0);
            z = p_ans.coeff(2, 0);
            q << x/z, y/z;
            mean_diff_ans += (ref_p - q).norm();

        }
        cout << "GT mean diff: " << mean_diff_ans/double(rows) << endl;
        cout << "threshold: " << thresh << ", " << "mean diff(my solution): " << mean_diff/double(rows) << endl;
        cout << C << endl;
    }

    return 0;
}