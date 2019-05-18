//
// Created by eight on 15/05/19.
//

#ifndef PROJET_ANNUEL_MATRIX_H
#define PROJET_ANNUEL_MATRIX_H


class Matrix {

public:
    double *getValues();
    double getColumns();
    double getRows();

    Matrix(double *values, int columns, int rows);

    void setValues(double *values);

    void setColumns(int columns);

    void setRows(int rows);

private:
    double *values;
    int columns;
    int rows;

};


#endif //PROJET_ANNUEL_MATRIX_H
