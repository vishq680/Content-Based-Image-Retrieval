/*
    * mathcing.cpp
    *
    *  Created on: Feb 05, 2024
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    Pupose: This C++ file contains the implementation of the functions for matching images using different feature extraction methods.

*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "matching.h"
#include <numeric>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <filesystem>

// Function: computeBaselineFeatures
// Input: cv::Mat &img - a reference to an OpenCV matrix representing an image
// Output: std::vector<double> - a vector of double values representing the features of the image
// This function computes the baseline features of an image by extracting a 7x7 square from the center of the image and converting it into a 1D vector.
std::vector<double> computeBaselineFeatures(const cv::Mat &img)
{
    // Compute the center of the image
    cv::Point center(img.cols / 2, img.rows / 2);

    // Define the size of the square
    int squareSize = 7;

    // Compute the top left corner of the square
    cv::Point topLeft(center.x - squareSize / 2, center.y - squareSize / 2);

    // Extract the 7x7 square from the center of the image
    cv::Mat square = img(cv::Rect(topLeft.x, topLeft.y, squareSize, squareSize));

    // Convert the 7x7 square into a 1D vector
    std::vector<double> features;
    for (int i = 0; i < square.rows; ++i)
    {
        for (int j = 0; j < square.cols; ++j)
        {
            // Push each pixel value into the features vector
            features.push_back(square.at<uchar>(i, j));
        }
    }

    // Return the 1D vector
    return features;
}

// Function: compareBaselineImages
// Input: const std::vector<double> &features1, const std::vector<double> &features2 - two vectors of double values representing the features of two images
// Output: double - the sum-of-squared-difference between the two feature vectors
// This function compares two images by computing the sum-of-squared-difference between their feature vectors.
double compareBaselineImages(const std::vector<double> &features1, const std::vector<double> &features2)
{
    // Check if the two feature vectors have the same size
    if (features1.size() != features2.size())
    {
        throw std::invalid_argument("The two feature vectors must have the same size.");
    }

    // Compute the sum-of-squared-difference between the two feature vectors
    double distance = 0.0;
    for (size_t i = 0; i < features1.size(); ++i)
    {
        double diff = features1[i] - features2[i];
        distance += diff * diff;
    }

    // Return the computed distance
    return distance;
}

// Function: computeColorHistogram
// Input: const cv::Mat &image - a reference to an OpenCV matrix representing an image
// Output: std::vector<double> - a vector of double values representing the color histogram of the image
// This function computes the color histogram of an image by counting the number of pixels of each color and normalizing the counts.
std::vector<double> computeColorHistogram(const cv::Mat &image)
{
    int bins = 16;
    std::vector<double> histogram(bins * bins, 0);

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            // Get the color of the pixel at (y, x)
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);

            // Normalize the red, green, and blue components of the color
            double r = color[2] / 255.0; // normalize red
            double g = color[1] / 255.0; // normalize green
            double b = color[0] / 255.0; // normalize blue

            // Compute the sum of the normalized components
            double sum = r + g + b + std::numeric_limits<double>::epsilon();

            // Compute the rg chromaticity
            int rIndex = static_cast<int>((r / sum) * (bins - 1)); // rg chromaticity
            int gIndex = static_cast<int>((g / sum) * (bins - 1)); // rg chromaticity

            // Compute the index in the histogram for the current color
            int index = rIndex * bins + gIndex;

            // Increment the count for the current color in the histogram
            histogram[index]++;
        }
    }

    // Return the computed histogram
    return histogram;
}

// Function: computeColorHistogram2
// Input: const cv::Mat &image - a reference to an OpenCV matrix representing an image
// Output: std::vector<double> - a vector of double values representing the color histogram of the image
// This function computes the color histogram of an image by counting the number of pixels of each color and normalizing the counts.
std::vector<double> computeColorHistogram2(const cv::Mat &image)
{
    int bins = 8;
    std::vector<double> histogram(bins * bins * bins, 0);

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            int rIndex = color[2] / (256 / bins); // Scale down the red value
            int gIndex = color[1] / (256 / bins); // Scale down the green value
            int bIndex = color[0] / (256 / bins); // Scale down the blue value
            int index = rIndex * bins * bins + gIndex * bins + bIndex;
            histogram[index]++;
        }
    }

    // Normalize the histogram
    double sum = std::accumulate(histogram.begin(), histogram.end(), 0.0);
    for (double &bin : histogram)
    {
        bin /= sum;
    }

    return histogram;
}

// Function: histogramIntersectionDistance
// Input: const std::vector<double> &histogram1, const std::vector<double> &histogram2 - two vectors of double values representing two histograms
// Output: double - the intersection distance between the two histograms
// This function computes the intersection distance between two histograms by summing the minimum values for each bin.
double histogramIntersectionDistance(const std::vector<double> &histogram1, const std::vector<double> &histogram2)
{
    double distance = 0.0;

    for (int i = 0; i < histogram1.size(); ++i)
    {
        distance += std::min(histogram1[i], histogram2[i]);
    }

    return distance;
}

// Function: euclideanDistance
// Input: const std::vector<double> &histogram1, const std::vector<double> &histogram2 - two vectors of double values representing two histograms
// Output: double - the Euclidean distance between the two histograms
// This function computes the Euclidean distance between two histograms by summing the squared differences for each bin and taking the square root of the result.
double euclideanDistance(const std::vector<double> &histogram1, const std::vector<double> &histogram2)
{
    double distance = 0.0;

    for (int i = 0; i < histogram1.size(); ++i)
    {
        double diff = histogram1[i] - histogram2[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

// Function: chiSquareDistance
// Input: const std::vector<double> &histogram1, const std::vector<double> &histogram2 - two vectors of double values representing two histograms
// Output: double - the Chi-square distance between the two histograms
// This function computes the Chi-square distance between two histograms by summing the squared differences for each bin divided by the sum of the bin values.
double chiSquareDistance(const std::vector<double> &histogram1, const std::vector<double> &histogram2)
{
    double distance = 0.0;

    for (int i = 0; i < histogram1.size(); ++i)
    {
        if (histogram1[i] + histogram2[i] > 0.0)
        {
            double diff = histogram1[i] - histogram2[i];
            distance += diff * diff / (histogram1[i] + histogram2[i]);
        }
    }

    return 0.5 * distance;
}

// Function: computeMultiHistogram
// Input: const cv::Mat &image - a reference to an OpenCV matrix representing an image
// Output: std::vector<std::vector<double>> - a vector of vectors of double values representing the multi-histogram of the image
// This function computes the multi-histogram of an image by computing the color histogram for the whole image and for the center of the image.
std::vector<std::vector<double>> computeMultiHistogram(const cv::Mat &image)
{
    int bins = 8;
    std::vector<std::vector<double>> histograms(2, std::vector<double>(bins * bins * bins, 0));

    // Compute whole-image histogram
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            int rIndex = color[2] / (256 / bins); // Scale down the red value
            int gIndex = color[1] / (256 / bins); // Scale down the green value
            int bIndex = color[0] / (256 / bins); // Scale down the blue value
            int index = rIndex * bins * bins + gIndex * bins + bIndex;
            histograms[0][index]++;
        }
    }

    // Compute center-image histogram
    cv::Rect region(image.cols / 4, image.rows / 4, image.cols / 2, image.rows / 2);
    cv::Mat subImage = image(region);
    for (int y = 0; y < subImage.rows; ++y)
    {
        for (int x = 0; x < subImage.cols; ++x)
        {
            cv::Vec3b color = subImage.at<cv::Vec3b>(y, x);
            int rIndex = color[2] / (256 / bins); // Scale down the red value
            int gIndex = color[1] / (256 / bins); // Scale down the green value
            int bIndex = color[0] / (256 / bins); // Scale down the blue value
            int index = rIndex * bins * bins + gIndex * bins + bIndex;
            histograms[1][index]++;
        }
    }

    return histograms;
}

// Function: multiHistogramIntersectionDistance
// Input: const std::vector<std::vector<double>> &histograms1, const std::vector<std::vector<double>> &histograms2 - two vectors of vectors of double values representing two multi-histograms
// Output: double - the intersection distance between the two multi-histograms
// This function computes the intersection distance between two multi-histograms by summing the intersection distances for each histogram with different weights.
double multiHistogramIntersectionDistance(const std::vector<std::vector<double>> &histograms1, const std::vector<std::vector<double>> &histograms2)
{
    double distance = 0.0;

    for (int i = 0; i < histograms1.size(); ++i)
    {
        double intersection = 0.0;

        for (int j = 0; j < histograms1[i].size(); ++j)
        {
            intersection += std::min(histograms1[i][j], histograms2[i][j]);
        }

        double weight = (i == 0) ? 0.5 : 1.0; // Use different weights for the whole-image and center-image histograms
        distance += weight * intersection;
    }

    return distance;
}

// Function: multiHistogramIntersectionDistance2
// Input: const std::vector<std::vector<double>> &histograms1, const std::vector<std::vector<double>> &histograms2 - two vectors of vectors of double values representing two multi-histograms
// Output: double - the average intersection distance between the two multi-histograms
// This function computes the average intersection distance between two multi-histograms by summing the intersection distances for each histogram and dividing by the number of histograms.
double multiHistogramIntersectionDistance2(const std::vector<std::vector<double>> &histograms1, const std::vector<std::vector<double>> &histograms2)
{
    double distance = 0.0;

    for (int i = 0; i < histograms1.size(); ++i)
    {
        double intersection = 0.0;

        for (int j = 0; j < histograms1[i].size(); ++j)
        {
            intersection += std::min(histograms1[i][j], histograms2[i][j]);
        }

        distance += intersection; // Unweighted averaging
    }

    return distance / histograms1.size(); // Divide by the number of histograms to get the average
}

// Function to apply sobel filter in X direction
/*
Arguments:
- src: Source image
- dst: Destination image, after applying the sobel filter
Returns:
- 0 on success
*/
int sobelX3x3(const cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is empty
    if (src.empty())
    {
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Define the 3x3 Sobel kernel for the X direction
    int kernel[3] = {-1, 0, 1};

    // Loop over each pixel in the image, excluding the border
    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            // Initialize the sum for each color channel
            cv::Vec3s sum = cv::Vec3s(0, 0, 0);

            // Convolve with the 3x3 Sobel filter in the horizontal direction
            for (int kx = -1; kx <= 1; kx++)
            {
                cv::Vec3b color = src.at<cv::Vec3b>(y, x + kx);
                int weight = kernel[kx + 1];

                // Apply the filter to each color channel
                sum[0] += weight * color[0];
                sum[1] += weight * color[1];
                sum[2] += weight * color[2];
            }

            // Assign the result to the destination image
            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }

    return 0;
}

// Function to apply sobel filter in Y direction
/*
Arguments:
- src: Source image
- dst: Destination image, after applying the sobel filter
Returns:
- 0 on success
*/
int sobelY3x3(const cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    int kernel[3] = {-1, 0, 1};

    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            cv::Vec3s sum = cv::Vec3s(0, 0, 0);

            for (int ky = -1; ky <= 1; ky++)
            {
                cv::Vec3b color = src.at<cv::Vec3b>(y + ky, x);
                int weight = kernel[ky + 1];

                sum[0] += weight * color[0];
                sum[1] += weight * color[1];
                sum[2] += weight * color[2];
            }

            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }

    return 0;
}

// Function: computeHistogram
// Input: cv::Mat &image - a reference to an OpenCV matrix representing an image
// Output: std::vector<double> - a vector of double values representing the histogram of the image
// This function computes the histogram of an image by counting the number of pixels of each intensity level.
std::vector<double> computeHistogram(const cv::Mat &image)
{
    std::vector<double> histogram(256, 0);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            histogram[(int)image.at<uchar>(i, j)]++;
        }
    }

    // Normalize the histogram
    for (int i = 0; i < 256; i++)
    {
        histogram[i] /= (image.rows * image.cols);
    }

    return histogram;
}

// Function to compute the magnitude of the gradient of an image
/*
Arguments:
- src: Source image
- dst: Destination image, after calculating the gradient magnitude
Returns:
- 0 on success
*/
int magnitude(const cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    cv::Mat sx, sy;

    // Create intermediate images for Sobel gradients in X and Y directions
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);

    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++)
    {
        for (int x = 0; x < sx.cols; x++)
        {
            // Extract color components from Sobel gradient images
            cv::Vec3s colorX = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s colorY = sy.at<cv::Vec3s>(y, x);

            cv::Vec3b &colorDst = dst.at<cv::Vec3b>(y, x);

            // Calculate the magnitude of the gradient for each color channel
            for (int i = 0; i < 3; i++)
            {
                float mag = std::sqrt(colorX[i] * colorX[i] + colorY[i] * colorY[i]);
                colorDst[i] = cv::saturate_cast<uchar>(mag);
            }
        }
    }

    return 0;
}

// Function to compute the color and texture feature vector for an image
/*
Arguments:
- image: Source image
Returns:
- std::vector<double>: The feature vector
*/
std::vector<double> computeColorAndTextureFeatures(const cv::Mat &image)
{
    // Compute the color histogram
    std::vector<double> colorHistogram = computeColorHistogram(image);

    // Compute the Sobel magnitude image
    cv::Mat sobelMagnitude;
    magnitude(image, sobelMagnitude); // Fix: Pass 'src' and 'sobelMagnitude' as arguments to the 'magnitude' function.

    // Compute the texture histogram
    std::vector<double> textureHistogram = computeHistogram(sobelMagnitude);

    // Concatenate the color histogram and the texture histogram
    std::vector<double> featureVector;
    featureVector.insert(featureVector.end(), colorHistogram.begin(), colorHistogram.end());
    featureVector.insert(featureVector.end(), textureHistogram.begin(), textureHistogram.end());

    return featureVector;
}

// Function to compute the distance between two feature vectors
/*
Arguments:
- features1: The first feature vector
- features2: The second feature vector
Returns:
- double: The distance between the two feature vectors
*/
double computeFeatureDistance(const std::vector<double> &features1, const std::vector<double> &features2)
{
    // Compute the Euclidean distance between the two feature vectors
    double distance = 0.0;
    for (size_t i = 0; i < features1.size(); i++)
    {
        double diff = features1[i] - features2[i];
        distance += diff * diff;
    }
    distance = std::sqrt(distance);

    return distance;
}

std::vector<double> computeCoOccurrenceMatrixFeatures(const cv::Mat &image)
{
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat coOccurrenceMatrix;
    int histSize = 256;       // Histogram size for each dimension
    float range[] = {0, 256}; // Range of intensity values
    const float *histRange = {range};
    cv::calcHist(&grayImage, 1, 0, cv::Mat(), coOccurrenceMatrix, 1, &histSize, &histRange);

    double energy = 0.0;
    double entropy = 0.0;

    for (int i = 0; i < coOccurrenceMatrix.rows; i++)
    {
        float value = coOccurrenceMatrix.at<float>(i);
        energy += value * value;
        if (value > 0)
        {
            entropy -= value * log2(value);
        }
    }

    return {energy, entropy};
}

std::vector<double> computeFourierTransformFeatures(const cv::Mat &image)
{
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat padded;
    int m = cv::getOptimalDFTSize(grayImage.rows);
    int n = cv::getOptimalDFTSize(grayImage.cols);
    cv::copyMakeBorder(grayImage, padded, 0, m - grayImage.rows, 0, n - grayImage.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);

    cv::dft(complexImage, complexImage);

    cv::split(complexImage, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magImage = planes[0];

    magImage += cv::Scalar::all(1);
    cv::log(magImage, magImage);

    cv::normalize(magImage, magImage, 0, 1, cv::NORM_MINMAX);

    cv::resize(magImage, magImage, cv::Size(16, 16));

    std::vector<double> features;
    features.reserve(16 * 16);
    for (int i = 0; i < magImage.rows; i++)
    {
        for (int j = 0; j < magImage.cols; j++)
        {
            features.push_back(magImage.at<float>(i, j));
        }
    }

    return features;
}

// Function to read CSV file
std::unordered_map<std::string, ImageData> readCSV(const std::string &filename)
{
    std::unordered_map<std::string, ImageData> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        ImageData imgData;
        std::getline(ss, imgData.filename, ',');
        std::string feature;
        while (std::getline(ss, feature, ','))
        {
            imgData.features.push_back(std::stod(feature));
        }
        data[imgData.filename] = imgData;
    }
    return data;
}

// Function to calculate cosine distance
double cosineDistance(const std::vector<double> &v1, const std::vector<double> &v2)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0u; i < v1.size(); ++i)
    {
        dot += v1[i] * v2[i];
        denom_a += v1[i] * v1[i];
        denom_b += v2[i] * v2[i];
    }
    return 1.0 - (dot / (sqrt(denom_a) * sqrt(denom_b)));
}

std::vector<double> getFeatureVector(const std::unordered_map<std::string, ImageData> &data, const std::string &targetFilename)
{
    std::string filename = targetFilename.substr(targetFilename.find_last_of("\\/") + 1);

    if (data.find(filename) != data.end())
    {
        // The filename exists in the map
        return data.at(filename).features;
    }
    else
    {
        // The filename does not exist in the map
        std::cerr << "Error: The filename " << filename << " does not exist in the data map." << std::endl;
        return std::vector<double>();
    }
}

// Function to match the embedding vectors for two images
double matchImages(const std::unordered_map<std::string, ImageData> &data, const std::string &filename1, const std::string &filename2)
{
    std::vector<double> features1 = getFeatureVector(data, filename1);
    std::vector<double> features2 = getFeatureVector(data, filename2);
    return cosineDistance(features1, features2);
}
