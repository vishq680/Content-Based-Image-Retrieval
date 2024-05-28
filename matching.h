/*
    * mathcing.cpp
    *
    *  Created on: Feb 05, 2024
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    Pupose: This file contains the function declarations for matching.cpp

*/
#ifndef MATCHING_H
#define MATCHING_H

#include <string>
#include <vector>

// Structure to hold image features and filename
struct ImageFeatures
{
    std::string filename;
    std::vector<double> features;
};

// Function to compute baseline image features
std::vector<double> computeBaselineFeatures(const cv::Mat &img);

// Function to compare two baseline images
double compareBaselineImages(const std::vector<double> &features1, const std::vector<double> &features2);

// Function to compute color histogram of an image
std::vector<double> computeColorHistogram(const cv::Mat &image);

// Function to compute histogram intersection distance between two histograms
double histogramIntersectionDistance(const std::vector<double> &histogram1, const std::vector<double> &histogram2);

// Function to compute color histogram of an image (alternative method)
std::vector<double> computeColorHistogram2(const cv::Mat &image);

// Function to compute multiple histograms of an image
std::vector<std::vector<double>> computeMultiHistogram(const cv::Mat &image);

// Function to compute multi-histogram intersection distance between two sets of histograms
double multiHistogramIntersectionDistance(const std::vector<std::vector<double>> &histograms1, const std::vector<std::vector<double>> &histograms2);

// Function to compute Euclidean distance between two histograms
double euclideanDistance(const std::vector<double> &histogram1, const std::vector<double> &histogram2);

// Function to compute Chi-Square distance between two histograms
double chiSquareDistance(const std::vector<double> &histogram1, const std::vector<double> &histogram2);

// Function to compute multi-histogram intersection distance between two sets of histograms (alternative method)
double multiHistogramIntersectionDistance2(const std::vector<std::vector<double>> &histograms1, const std::vector<std::vector<double>> &histograms2);

// Function to apply Sobel X operator on an image
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

// Function to apply Sobel Y operator on an image
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Function to compute magnitude of gradients in an image
int magnitude(cv::Mat &src, cv::Mat &dst);

// Function to compute color and texture features of an image
std::vector<double> computeColorAndTextureFeatures(const cv::Mat &image);

// Function to compute distance between two sets of features
double computeFeatureDistance(const std::vector<double> &features1, const std::vector<double> &features2);

// Function to compute co-occurrence matrix features of an image
std::vector<double> computeCoOccurrenceMatrixFeatures(const cv::Mat &image);

// Function to compute Fourier Transform features of an image
std::vector<double> computeFourierTransformFeatures(const cv::Mat &image);

// Function to compute histogram of an image
std::vector<double> computeHistogram(const cv::Mat &image);

// Structure to hold image data
struct ImageData
{
    std::string filename;
    std::vector<double> features;
};

// Function to read CSV file and return a map of filenames to ImageData
std::unordered_map<std::string, ImageData> readCSV(const std::string &filename);

// Function to compute cosine distance between two vectors
double cosineDistance(const std::vector<double> &v1, const std::vector<double> &v2);

// Function to get feature vector for a given filename from a map of filenames to ImageData
std::vector<double> getFeatureVector(const std::unordered_map<std::string, ImageData> &data, const std::string &filename);

// Function to match two images and return the distance between their feature vectors
double matchImages(const std::unordered_map<std::string, ImageData> &data, const std::string &filename1, const std::string &filename2);

#endif // MATCHING_H