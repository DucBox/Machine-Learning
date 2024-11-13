#include <iostream>
#include <iostream>
#include "utils.h"
#include <string>
#include <sstream>
#include <cmath>
#include <omp.h>
using namespace std;


void create_histogram(int *hist, int **img, int num_rows, int num_cols){
    int  smallMatrix[3][3];
    int i = 1;
    int decimal = 0;

    while ( i <= num_rows) {
        int j = 1;
        while ( j <= num_cols) {
            if (img[i][j] <= img[i - 1][j - 1]) {
                smallMatrix[0][0] = 0;
                //cout << img[i][j] << " " << endl;
            }
            else{
                smallMatrix[0][0] = 1;
                // cout << img[i][j] << " " << endl;
            }
            if (img[i][j] <= img[i - 1][j]) {
                smallMatrix[0][1] = 0;
            }
            else  {
                smallMatrix[0][1] = 1;

            }
            if (img[i][j] <= img[i - 1][j + 1]) {
                smallMatrix[0][2] = 0;
            }
            else {
                smallMatrix[0][2] = 1;
            }
            if (img[i][j] <= img[i][j - 1]) {
                smallMatrix[1][0] = 0;
            }
            else {
                smallMatrix[1][0] = 1;
            }
            if (img[i][j] <= img[i][j + 1]) {
                smallMatrix[1][2] = 0;
            }
            else  {
                smallMatrix[1][2] = 1;
            }
            if (img[i][j] <= img[i + 1][j - 1]) {
                smallMatrix[2][0] = 0;
            }
            else {
                smallMatrix[2][0] = 1;
            }
            if (img[i][j] <= img[i + 1][j]) {
                smallMatrix[2][1] = 0;
            }
            else {
                smallMatrix[2][1] = 1;
            }
            if (img[i][j] <= img[i + 1][j + 1]) {
                smallMatrix[2][2] = 0;
            }
            else {
                smallMatrix[2][2] = 1;
            }
            decimal = smallMatrix[0][0] * int(pow(2, 7)) + smallMatrix[0][1] * int(pow(2, 6)) + smallMatrix[0][2] * int(pow(2, 5)) +
                      smallMatrix[1][2] * int(pow(2, 4)) +
                      smallMatrix[2][2] * int(pow(2, 3)) + smallMatrix[2][1] * int(pow(2, 2)) + smallMatrix[2][0] * int(pow(2, 1)) +
                      smallMatrix[1][0] * 1;

            hist[decimal]++;
            // cout <<  hist[decimal] << " " ;
            j++;
        }
        i++;
    }
    }

// Function to initialize the training set to zero
void initialize_training_set(int*** training_set, int nrOfIds, int nrOfPhotosPerId, int histogramSize) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < nrOfIds; i++) {
        for (int j = 0; j < nrOfPhotosPerId; j++) {
            for (int e = 0; e < histogramSize; e++) {
                training_set[i][j][e] = 0;
            }
        }
    }
}

void process_images(int*** training_set, int nrOfIds, int nrOfPhotosPerId, int num_rows, int num_cols) {
    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            int num_threads = omp_get_num_threads();
            cout << "Số luồng đang chạy: " << num_threads << endl;
        }

        #pragma omp for collapse(2)
        for (int w = 1; w <= nrOfIds; w++) {
            for (int q = 1; q <= nrOfPhotosPerId; q++) {
                string filename = "data_images/" + to_string(w) + "." + to_string(q) + ".txt";

                int **image = read_pgm_file(filename, num_rows, num_cols);
                int **img = alloc_2d_matrix(num_rows + 2, num_cols + 2);  // Enhanced image matrix

                // Khởi tạo ma trận img với 0 (tuần tự)
                for (int i = 0; i < num_rows + 2; i++) {
                    for (int j = 0; j < num_cols + 2; j++) {
                        img[i][j] = 0;
                    }
                }

                // Sao chép dữ liệu từ image vào img (tuần tự)
                for (int i = 1; i <= num_rows; i++) {
                    for (int j = 1; j <= num_cols; j++) {
                        img[i][j] = image[i - 1][j - 1];
                    }
                }

                // Tạo histogram (tuần tự)
                create_histogram(training_set[w - 1][q - 1], img, num_rows, num_cols);

                // Giải phóng bộ nhớ
                dealloc_2d_matrix(image, num_rows, num_cols);
                dealloc_2d_matrix(img, num_rows + 2, num_cols + 2);
            }
        }
    }
}
// Function to find the closest match using histogram comparison
int find_closest(int*** training_set, int num_persons, int num_training, int size, int* test_image) {
    double closestValue = 1e10;
    int closest = -1;

    for (int i = 0; i < num_persons; i++) {
        for (int j = 0; j < num_training; j++) {
            double dist = 0;
            for (int k = 0; k < size; k++) {
                if (training_set[i][j][k] + test_image[k] != 0) {
                    dist += 0.5 * pow(training_set[i][j][k] - test_image[k], 2) /
                            (training_set[i][j][k] + test_image[k]);
                }
            }
            if (dist < closestValue) {
                closestValue = dist;
                closest = i;
            }
        }
    }
    return closest + 1;
}

// Main function
int main(int argc, char* argv[]) {
    int k = stoi(argv[1]); // Take k from argv
    int nrOfIds = 18;
    int nrOfPhotosPerId = 20;
    int num_rows = 200;
    int num_cols = 180;
    int histogramSize = 256;
    int*** training_set = new int**[nrOfIds];
    for (int i = 0; i < nrOfIds; i++) {
        training_set[i] = alloc_2d_matrix(nrOfPhotosPerId, histogramSize);
    }

    double start_parallel = omp_get_wtime();
    initialize_training_set(training_set, nrOfIds, nrOfPhotosPerId, histogramSize);
    // double end_parallel = omp_get_wtime();
    // double parallel_time_1 = (end_parallel - start_parallel) * 1000;

    // double start_parallel_2 = omp_get_wtime();
    process_images(training_set, nrOfIds, nrOfPhotosPerId, num_rows, num_cols);
    double end_parallel_2 = omp_get_wtime();
    double parallel_time_2 = (end_parallel_2 - start_parallel) * 1000;

    int tests = 0, correctIds = 0;
    for (int i = 0; i < nrOfIds; i++) {
        for (int j = k; j < nrOfPhotosPerId; j++) {
            int testResultId = find_closest(training_set, nrOfIds, k, histogramSize, training_set[i][j]);
            tests++;
            if (testResultId == i + 1) {
                correctIds++;
            }
        }
    }

    // cout << "Accuracy: " << correctIds << " correct answers for " << tests << " tests" << endl;
    // cout << "Parallel time initial training set: " << parallel_time_1 << " ms" << endl
    cout << "Parallel time: " << parallel_time_2 << " ms" << endl;

    for (int i = 0; i < nrOfIds; i++) {
        dealloc_2d_matrix(training_set[i], nrOfPhotosPerId, histogramSize);
    }
    delete[] training_set;

    return 0;
}
