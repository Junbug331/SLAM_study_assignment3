#include "utils.hpp"
#include <sstream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>

/*
Reference Code for loading csv data intn Eigen
https://aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
*/


Eigen::MatrixXd readCSVFiletoEigen(const std::string& path, bool skip_first_row)
{
    if (path.substr(path.size()-3) != "csv")
        throw std::runtime_error("input file is not csv file!\n");

    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Can't open file " + path + "\n");

    std::vector<double> matrix_entries;
    std::string matrix_row_str;
    std::string matrix_entry;
    int rows = 0; 

    matrix_entries.reserve(100);

    while (getline(in, matrix_row_str)) 
    {
        if (skip_first_row)
        {
            skip_first_row = false;
            continue;
        }
        std::stringstream matrix_row_str_stream(matrix_row_str);

        while (getline(matrix_row_str_stream, matrix_entry, ','))
            matrix_entries.push_back(std::stod(matrix_entry));
        rows++;
    }

    in.close();

    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrix_entries.data(), rows, matrix_entries.size()/rows);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> splitInputMatrix(const Eigen::MatrixXd& input_mat)
{
    int rows = input_mat.rows();
    Eigen::MatrixXd P_I = Eigen::MatrixXd::Zero(rows, 2);
    Eigen::MatrixXd P_M = Eigen::MatrixXd::Zero(rows, 3);

    for(int r=0; r<rows; ++r)
    {
        P_I.block<1, 2>(r, 0) = input_mat.block<1, 2>(r, 0);
        P_M.block<1, 3>(r, 0) = input_mat.block<1, 3>(r, 2);
    }
    return {P_I, P_M};
}

std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point3d>> readInputOpenCV(const std::string& path)
{
    if (path.substr(path.size()-3) != "csv")
        throw std::runtime_error("input file is not csv file!\n");

    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Can't open file " + path + "\n");

    std::vector<double> matrix_entries;
    std::string matrix_row_str;
    std::string matrix_entry;
    int cnt = 0;
    bool skip_first_row = true;

    matrix_entries.reserve(100);

    while (getline(in, matrix_row_str))
    {
        if (skip_first_row)
        {
            skip_first_row = false;
            continue;
        }

        std::stringstream matrix_row_str_stream(matrix_row_str);

        while (getline(matrix_row_str_stream, matrix_entry, ','))
            matrix_entries.push_back(std::stod(matrix_entry));

        cnt++;
    }

    std::vector<cv::Point2d> imgPoints;
    imgPoints.reserve(cnt);
    std::vector<cv::Point3d> objPoints;
    objPoints.reserve(cnt);

    for (int j=0; j<cnt; j++)
    {
        int i = j*5;
        imgPoints.emplace_back(cv::Point2f(matrix_entries[i], matrix_entries[i+1]));
        objPoints.emplace_back(cv::Point3f(matrix_entries[i+2], matrix_entries[i+3], matrix_entries[i+4]));
    }
    in.close();
    return {imgPoints, objPoints};
}

unsigned long long comb(int n, int r, std::vector<std::vector<unsigned long long>> &cache)
{
    if (cache[n][r]) return cache[n][r];
    else if (n == 0 || r == 0 || n == r) return cache[n][r] = 1ULL;
    else return cache[n][r] = comb(n-1, r-1, cache) + comb(n-1, r, cache);
}

unsigned long long comb(int n, int r)
{
    std::vector<std::vector<unsigned long long>> cache(100, std::vector<unsigned long long>(100, 0));
    return comb(n, r, cache);
}
