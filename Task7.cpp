/*
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    This code incorporates Gabor features for describing images and using Euclidean distance as the distance metric.


*/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;

// Function to print the contents of a feature vector
// Parameters:
//   - featureVector: Feature vector (cv::Mat) to be printed
// Return type: void
void printFeatureVector(const cv::Mat &featureVector)
{
    cout << "Feature Vector:" << endl;
    for (int i = 0; i < featureVector.rows; ++i)
    {
        for (int j = 0; j < featureVector.cols; ++j)
        {
            cout << featureVector.at<double>(i, j) << " ";
        }
        cout << endl;
    }
}

// Function to compute Gabor features for an input image
// Parameters:
//   - image: Input image (cv::Mat)
// Return type: cv::Mat (Gabor features)
cv::Mat computeGaborFeatures(const cv::Mat &image)
{
    if (image.empty())
    {
        // Return an empty cv::Mat to indicate an error
        return cv::Mat();
    }

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    if (gray.empty())
    {
        // Handle grayscale conversion failure
        cerr << "Error: Grayscale conversion failed." << endl;
        return cv::Mat();
    }

    // Gabor filter parameters
    int kernel_size = 31;
    double sigma = 4.0;
    double theta = CV_PI / 4.0;
    double lambda = 10.0;
    double gamma = 0.5;

    // Generate Gabor kernel
    cv::Mat gabor_kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma);

    // Apply Gabor filter to the grayscale image
    cv::Mat gabor_response;
    cv::filter2D(gray, gabor_response, CV_64F, gabor_kernel);

    // Extract statistical features from the Gabor response (e.g., mean, variance, etc.)
    cv::Scalar mean, stddev;
    cv::meanStdDev(gabor_response, mean, stddev);

    // Create a feature vector with mean value
    cv::Mat features(1, 1, CV_64F);
    features.at<double>(0, 0) = mean[0];

    return features;
}

// Function to read feature vectors from a CSV file
// Parameters:
//   - csvFilePath: Path to the CSV file containing feature vectors
// Return type: vector of pairs, where each pair contains a filename and its corresponding feature vector (cv::Mat)
vector<pair<string, cv::Mat>> readFeatureVectors(const string &csvFilePath)
{
    vector<pair<string, cv::Mat>> featureVectors;

    ifstream inputFile(csvFilePath);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open the CSV file." << endl;
        exit(1);
    }

    // Assuming a directory path for images
    string directoryPath = "../../olympus/";

    string line;
    while (getline(inputFile, line))
    {
        istringstream ss(line);
        string fileName;
        getline(ss, fileName, ',');

        // Read image from the specified directory
        cv::Mat image = cv::imread(directoryPath + fileName);

        if (image.empty())
        {
            cerr << "Error: Unable to read the image file " << fileName << endl;
            continue;
        }

        // Compute Gabor features for the image
        cv::Mat gaborFeatures = computeGaborFeatures(image);

        // Add filename and Gabor features to the vector
        featureVectors.push_back({fileName, gaborFeatures.clone()});
    }

    inputFile.close();
    return featureVectors;
}

// Function to compute Euclidean distance between two feature vectors
// Parameters:
//   - features1: First feature vector (cv::Mat)
//   - features2: Second feature vector (cv::Mat)
// Return type: double (distance)
double computeEuclideanDistance(const cv::Mat &features1, const cv::Mat &features2)
{
    cv::Mat diff;
    cv::absdiff(features1, features2, diff);
    cv::Mat squaredDiff = diff.mul(diff);
    double distance = cv::sqrt(cv::sum(squaredDiff)[0]);
    return distance;
}

// Main function
int main(int argc, char *argv[])
{
    // Check if the correct number of command-line arguments is provided
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <feature_vectors_csv> <N>" << endl;
        return 1;
    }

    string featureVectorsCsv = argv[1];
    int N = stoi(argv[2]);

    vector<pair<string, cv::Mat>> featureVectors = readFeatureVectors(featureVectorsCsv);

    // Check if feature vectors were successfully loaded
    if (featureVectors.empty())
    {
        cerr << "Error: No valid feature vectors were loaded." << endl;
        return 1;
    }

    // Read the target image
    cv::Mat targetImage = cv::imread("../banana.jpeg");
    
    cv::Mat targetFeatureVector = computeGaborFeatures(targetImage);

    // Perform distance computation and sorting similar to your original code
    vector<pair<string, double>> imageDistances;
    for (const auto &entry : featureVectors)
    {
        double distance = computeEuclideanDistance(targetFeatureVector, entry.second);
        // Add the image filename and distance to the vector
        imageDistances.push_back({entry.first, distance});
    }

    // Sort the distances and print the top N matches
    sort(imageDistances.begin(), imageDistances.end(), [](const pair<string, double> &a, const pair<string, double> &b)
         { return a.second < b.second; });

    for (int i = 0; i < N && i < imageDistances.size(); ++i)
    {
        cout << "Match " << i + 1 << ": " << imageDistances[i].first << " - Distance: " << imageDistances[i].second << endl;
    }

    // Return 0 to indicate successful execution
    return 0;
}
