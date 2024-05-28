/*
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    The purpose of this code is to perform content-based image retrieval using feature vectors. 
    The code allows users to find similar images to a given target image based on either the sum-of-square distance or cosine distance metric.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;

// Function to read feature vectors from a CSV file
// Parameters:
//   - csvFilePath: Path to the CSV file containing feature vectors
// Return type: vector of pairs, where each pair contains a filename and its corresponding feature vector (cv::Mat)
std::vector<std::pair<std::string, cv::Mat>> readFeatureVectors(const std::string &csvFilePath)
{
    // Vector to store pairs of filenames and feature vectors
    std::vector<std::pair<std::string, cv::Mat>> featureVectors;

    // Open the CSV file
    std::ifstream inputFile(csvFilePath);
    if (!inputFile.is_open())
    {
        std::cerr << "Error: Unable to open the CSV file." << std::endl;
        exit(1);
    }

    // Read each line in the CSV file
    std::string line;
    while (std::getline(inputFile, line))
    {
        // Parse the line using stringstream
        std::istringstream ss(line);
        std::string fileName;
        std::getline(ss, fileName, ',');

        // Vector to store double values from the CSV
        std::vector<double> values;
        std::string value;
        while (std::getline(ss, value, ','))
        {
            values.push_back(std::stod(value));
        }

        // Check if the size of values is not equal to 512
        if (values.size() != 512)
        {
            std::cerr << "Error: Unexpected number of values for feature vector in the CSV file." << std::endl;
            exit(1);
        }

        // Create a cv::Mat for the feature vector
        cv::Mat featureVector(values.size(), 1, CV_32F);
        for (int i = 0; i < values.size(); ++i)
        {
            featureVector.at<float>(i, 0) = values[i];
        }

        // Add the filename and feature vector to the vector
        featureVectors.push_back({fileName, featureVector.clone()});
    }

    // Close the file and return the vector
    inputFile.close();
    return featureVectors;
}

// Function to compute the sum-of-square distance between two feature vectors
// Parameters:
//   - features1: First feature vector (cv::Mat)
//   - features2: Second feature vector (cv::Mat)
// Return type: double (distance)
double computeSumSquareDistance(const cv::Mat &features1, const cv::Mat &features2)
{
    // Calculate the absolute difference and square it
    cv::Mat diff;
    cv::absdiff(features1, features2, diff);
    cv::Mat squaredDiff = diff.mul(diff);

    // Sum the squared differences and return the result
    double distance = cv::sum(squaredDiff)[0];
    return distance;
}

// Function to compute the cosine distance between two feature vectors
// Parameters:
//   - features1: First feature vector (cv::Mat)
//   - features2: Second feature vector (cv::Mat)
// Return type: double (distance)
double computeCosineDistance(const cv::Mat &features1, const cv::Mat &features2)
{
    // Normalize the feature vectors
    cv::Mat normalizedFeatures1, normalizedFeatures2;
    cv::normalize(features1, normalizedFeatures1);
    cv::normalize(features2, normalizedFeatures2);

    // Compute the cosine similarity and convert to distance
    double cosineSimilarity = normalizedFeatures1.dot(normalizedFeatures2);
    return 1.0 - cosineSimilarity;
}

// Main function
int main(int argc, char *argv[])
{
    // Check if the correct number of command-line arguments is provided
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_vectors_csv> <distance_metric> <N>" << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string targetImagePath = argv[1];
    std::string featureVectorsCsv = argv[2];
    std::string distanceMetric = argv[3];
    int N = std::stoi(argv[4]);

    // Read the target image
    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty())
    {
        std::cerr << "Error: Unable to read the target image." << std::endl;
        return 1;
    }

    // Read feature vectors from the CSV file
    std::vector<std::pair<std::string, cv::Mat>> featureVectors = readFeatureVectors(featureVectorsCsv);

    // Find the feature vector for the target image
    cv::Mat targetFeatureVector;
    std::string targetImageFileName = targetImagePath.substr(targetImagePath.find_last_of("/\\") + 1);
    for (const auto &pair : featureVectors)
    {
        if (pair.first == targetImageFileName)
        {
            targetFeatureVector = pair.second;
            break;
        }
    }

    // Check if the feature vector for the target image is found
    if (targetFeatureVector.empty())
    {
        std::cerr << "Error: Feature vector not found for the target image." << std::endl;
        return 1;
    }

    // Loop over the feature vectors to compute distances
    std::vector<std::pair<std::string, double>> imageDistances;
    for (const auto &entry : featureVectors)
    {
        if (entry.first != targetImagePath)
        {
            double distance = 0.0;

            // Choose the distance metric based on the input
            if (distanceMetric == "sum_square")
            {
                distance = computeSumSquareDistance(targetFeatureVector, entry.second);
            }
            else if (distanceMetric == "cosine")
            {
                distance = computeCosineDistance(targetFeatureVector, entry.second);
            }
            else
            {
                std::cerr << "Error: Unknown distance metric." << std::endl;
                return 1;
            }

            // Add the image filename and distance to the vector
            imageDistances.push_back({entry.first, distance});
        }
    }

    // Sort the list of matches based on distance
    std::sort(imageDistances.begin(), imageDistances.end(), [](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b)
              { return a.second < b.second; });

    // Print the top N matches
    for (int i = 0; i < N && i < imageDistances.size(); ++i)
    {
        std::cout << "Match " << i + 1 << ": " << imageDistances[i].first << " - Distance: " << imageDistances[i].second << std::endl;
    }

    // Return 0 to indicate successful execution
    return 0;
}
