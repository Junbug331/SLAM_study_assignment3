#pragma once
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <Eigen/Dense>
#include <filesystem>
#include <string>
#include <tuple>
#include <opencv2/core/mat.hpp>

Eigen::MatrixXd readCSVFiletoEigen(const std::string& path, bool skip_first_row=false);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> splitInputMatrix(const Eigen::MatrixXd& input_mat);
std::tuple<std::vector<cv::Point2d>, std::vector<cv::Point3d>> readInputOpenCV(const std::string& path);
unsigned long long comb(int n, int r, std::vector<std::vector<unsigned long long>> &cache);
unsigned long long comb(int n, int r);


#endif
