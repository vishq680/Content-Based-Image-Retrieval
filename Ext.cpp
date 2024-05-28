/*
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

This code performs content-based image retrieval by considering edge histogram features. 
The cosine distance is used as a similarity measure between feature vectors.

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

// Function to compute edge histogram features for an input image
// Parameters:
//   - image: Input image (cv::Mat)
// Return type: cv::Mat (edge histogram features)
cv::Mat computeEdgeHistogramFeatures(const cv::Mat &image)
{
    if (image.empty())
    {
        cerr << "Error: Input image is empty." << endl;
        return cv::Mat();
    }

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    if (gray.empty())
    {
        cerr << "Error: Grayscale conversion failed." << endl;
        return cv::Mat();
    }

    // Perform edge detection (you may need to fine-tune parameters based on your requirements)
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    // Compute histogram of edges
    cv::Mat hist;
    int histSize = 256; // Adjust as needed
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::calcHist(&edges, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // Convert histogram to feature vector
    cv::Mat features;
    hist.reshape(1, 1).convertTo(features, CV_64F);

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

        // Compute edge histogram features for the image
        cv::Mat edgeHistogramFeatures = computeEdgeHistogramFeatures(image);

        // Add filename and edge histogram features to the vector
        featureVectors.push_back({fileName, edgeHistogramFeatures.clone()});
    }

    inputFile.close();
    return featureVectors;
}

// Function to compute cosine distance between two feature vectors
// Parameters:
//   - features1: First feature vector (cv::Mat)
//   - features2: Second feature vector (cv::Mat)
// Return type: double (distance)
double computeCosineDistance(const cv::Mat &features1, const cv::Mat &features2)
{
    cv::Mat normalizedFeatures1, normalizedFeatures2;
    cv::normalize(features1, normalizedFeatures1);
    cv::normalize(features2, normalizedFeatures2);

    double cosineSimilarity = normalizedFeatures1.dot(normalizedFeatures2);
    return 1.0 - cosineSimilarity;
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

    // Read feature vectors from the CSV file
    vector<pair<string, cv::Mat>> featureVectors = readFeatureVectors(featureVectorsCsv);

    // Check if feature vectors were successfully loaded
    if (featureVectors.empty())
    {
        cerr << "Error: No valid feature vectors were loaded." << endl;
        return 1;
    }

    // Read the target image
    cv::Mat targetImage = cv::imread("../banana.jpeg");
    
    cv::Mat targetFeatureVector = computeEdgeHistogramFeatures(targetImage);

    // Perform distance computation and sorting similar to your original code
    vector<pair<string, double>> imageDistances;
    for (const auto &entry : featureVectors)
    {
        double distance = computeCosineDistance(targetFeatureVector, entry.second);

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
