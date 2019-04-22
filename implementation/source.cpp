#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT
#endif
extern "C" {

SUPEREXPORT double *create_linear_model(int inputCountPerSample) {
    auto W = new double[inputCountPerSample + 1];
    // TODO : initialisation random [-1,1]
    return W;
}

SUPEREXPORT void fit_classification_rosenblatt_rule(
        double *W,
        double *XTrain,
        int sampleCount,
        int inputCountPerSample,
        double *YTrain,
        double alpha, // Learning Rate
        int epochs // Nombre d'itération
) {
    for (auto i = 0; i < epochs; i++) {
        for (auto k = 0; k < sampleCount; k++) {
            // TODO : entrainement (correction des W, cf slides !)
        }
    }
}

SUPEREXPORT void fit_regression(
        double *W,
        double *XTrain,
        int SampleCount,
        int inputCountPerSample,
        double *YTrain
) {
    // TODO : entrainement (correction des W, cf slides !)
}

SUPEREXPORT double predict_regression(
        double *W,
        double *XToPredict,
        int inputCountPerSample
) {
    // TODO : Inférence (CF Slides !)
    return 0.42;
}

SUPEREXPORT double predict_classification(
        double *W,
        double *XToPredict,
        int inputCountPerSample
) {
    return predict_regression(W, XToPredict, inputCountPerSample) >= 0 ? 1.0 : -1.0;
}

SUPEREXPORT void delete_linear_model(double *W) {
    delete[] W;
}
}