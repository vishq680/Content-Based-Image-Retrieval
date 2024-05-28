/*
    * main.cpp
    *
    *  Created on: Feb 05, 2024
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    Pupose: This C++ program uses OpenCV to compare a target image with a set of images in a directory, based on a specified feature extraction method.
            It then sorts the images based on their similarity to the target image and prints the filenames of the top N matching images.

*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "matching.h"
#include <dirent.h>
#include <algorithm>
#include <filesystem>
using namespace std;

int main(int argc, char *argv[])
{
    // Check for sufficient arguments
    if (argc < 5)
    {
        printf("usage: %s <target filename> <directory path> <method> <number of images>\n", argv[0]);
        exit(-1);
    }

    // Define a structure to hold the filename and features of an image
    struct MultiHistogramFeatures
    {
        std::string filename;
        std::vector<std::vector<double>> features;
    };

    // Vector to hold multiple histogram features
    std::vector<MultiHistogramFeatures> multiHistogramFeatures;

    // Parse command line arguments
    std::string targetFilename = argv[1];
    std::string directory = argv[2];
    std::string method = argv[3];
    std::vector<std::vector<double>> targetMultiHistogram;

    int N = std::stoi(argv[4]);

    // Compute features for target image
    cv::Mat targetImage = cv::imread(targetFilename, cv::IMREAD_GRAYSCALE);
    std::vector<double> targetFeatures;
    std::unordered_map<std::string, ImageData> data;

    // Compute features for the target images based on the method provided
    if (method == "b")
    {
        targetFeatures = computeBaselineFeatures(targetImage);
    }

    else if (method == "h")
    {
        targetImage = cv::imread(targetFilename, cv::IMREAD_COLOR);
        targetFeatures = computeColorHistogram(targetImage);
    }

    else if (method == "h2")
    {
        targetImage = cv::imread(targetFilename, cv::IMREAD_COLOR);
        targetFeatures = computeColorHistogram2(targetImage);
    }

    else if (method == "m")
    {
        targetImage = cv::imread(targetFilename, cv::IMREAD_COLOR);
        targetMultiHistogram = computeMultiHistogram(targetImage);
    }
    else if (method == "t")
    {
        targetImage = cv::imread(targetFilename, cv::IMREAD_COLOR);
        targetFeatures = computeColorAndTextureFeatures(targetImage);
    }
    else if (method == "f")
    {
        targetImage = cv::imread(targetFilename, cv::IMREAD_COLOR);
        targetFeatures = computeFourierTransformFeatures(targetImage);
    }
    else if (method == "c")
    {
        targetImage = cv::imread(targetFilename, cv::IMREAD_COLOR);
        targetFeatures = computeCoOccurrenceMatrixFeatures(targetImage);
    }
    else if (method == "d")
    {
        // Read the target image's features from the CSV file
        std::string csvPath = "E:/MSCS/CVPR/projects/project2/data/ResNet18_olym.csv";
        data = readCSV(csvPath);

        targetFeatures = getFeatureVector(data, targetFilename);
        cout << "Target features read" << endl;
    }
    else
    {
        // Invalid method if the method is not one of the above
        printf("Invalid method\n");
        exit(-1);
    }

    // Loop over directory of images
    std::vector<ImageFeatures> imageFeatures;
    DIR *dirp;
    struct dirent *dp;
    dirp = opendir(directory.c_str());
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", directory.c_str());
        exit(-1);
    }

    while ((dp = readdir(dirp)) != NULL)
    {
        std::string filename(dp->d_name);
        std::string extension = filename.substr(filename.find_last_of(".") + 1);

        // Check if the file is an image
        if (extension == "jpg" || extension == "png" || extension == "ppm" || extension == "tif")
        {
            // Compute features for the input images based on the method provided

            if (method == "b")
            {
                ImageFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = computeBaselineFeatures(image);
                imageFeatures.push_back(img);
            }
            else if (method == "h")
            {
                ImageFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = computeColorHistogram(image);
                imageFeatures.push_back(img);
            }
            else if (method == "h2")
            {
                ImageFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = computeColorHistogram2(image);
                imageFeatures.push_back(img);
            }
            else if (method == "m")
            {
                MultiHistogramFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = computeMultiHistogram(image);
                multiHistogramFeatures.push_back(img);
            }
            else if (method == "t")
            {
                ImageFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = computeColorAndTextureFeatures(image);
                imageFeatures.push_back(img);
            }
            else if (method == "f")
            {
                ImageFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = computeFourierTransformFeatures(image);
                imageFeatures.push_back(img);
            }
            else if (method == "c")
            {
                ImageFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = computeCoOccurrenceMatrixFeatures(image);
                imageFeatures.push_back(img);
            }
            else if (method == "d")
            {
                ImageFeatures img;
                img.filename = directory + "/" + filename;
                cv::Mat image = cv::imread(img.filename, cv::IMREAD_COLOR);
                img.features = getFeatureVector(data, img.filename);
                imageFeatures.push_back(img);
            }
        }
    }
    closedir(dirp);

    // Sort images by their distance to the target image
    if (method == "m")
    {
        std::sort(multiHistogramFeatures.begin(), multiHistogramFeatures.end(),
                  [&targetMultiHistogram](const MultiHistogramFeatures &a, const MultiHistogramFeatures &b)
                  {
                      double distanceA = multiHistogramIntersectionDistance(targetMultiHistogram, a.features);
                      double distanceB = multiHistogramIntersectionDistance(targetMultiHistogram, b.features);
                      return distanceA < distanceB;
                  });

        // Print filenames of the top N matching images
        for (int i = 0; i < N && i < multiHistogramFeatures.size(); ++i)
        {
            std::cout << multiHistogramFeatures[i].filename << std::endl;
        }
    }
    else if (method == "d")
    {
        std::sort(imageFeatures.begin(), imageFeatures.end(),
                  [&targetFeatures, &data, &targetFilename](const ImageFeatures &a, const ImageFeatures &b)
                  {
                      double distanceA = matchImages(data, targetFilename, a.filename);
                      double distanceB = matchImages(data, targetFilename, b.filename);
                      return distanceA < distanceB;
                  });

        // Print filenames of the top N matching images
        for (int i = 0; i < N && i < imageFeatures.size(); ++i)
        {
            std::cout << imageFeatures[i].filename << std::endl;
        }
    }
    else
    {

        std::sort(imageFeatures.begin(), imageFeatures.end(),
                  [&targetFeatures, &method](const ImageFeatures &a, const ImageFeatures &b)
                  {
                      double distanceA, distanceB;
                      if (method == "b")
                      {
                          distanceA = compareBaselineImages(targetFeatures, a.features);
                          distanceB = compareBaselineImages(targetFeatures, b.features);
                      }
                      else if (method == "h" || method == "h2")
                      {
                          distanceA = euclideanDistance(targetFeatures, a.features);
                          distanceB = euclideanDistance(targetFeatures, b.features);
                      }
                      else if (method == "t" || method == "f" || method == "c")
                      {
                          distanceA = computeFeatureDistance(targetFeatures, a.features);
                          distanceB = computeFeatureDistance(targetFeatures, b.features);
                      }
                      // Add other matching methods here...
                      return distanceA < distanceB;
                  });

        // Print filenames of the top N matching images
        for (int i = 0; i < N && i < imageFeatures.size(); ++i)
        {
            std::cout << imageFeatures[i].filename << std::endl;
        }
    }

    return 0;
}