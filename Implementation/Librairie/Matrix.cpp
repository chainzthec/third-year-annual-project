//
// Created by eight on 15/05/19.
//

#include "Matrix.h"

double *Matrix::getValues() {
    return Matrix::values;
}

double Matrix::getColumns() {
    return Matrix::columns;
}

double Matrix::getRows() {
    return Matrix::rows;
}

void Matrix::setValues(double *values) {
    Matrix::values = values;
}

void Matrix::setColumns(int columns) {
    Matrix::columns = columns;
}

void Matrix::setRows(int rows) {
    Matrix::rows = rows;
}

Matrix::Matrix(double *values, int columns, int rows) : values(values), columns(columns), rows(rows) {}
