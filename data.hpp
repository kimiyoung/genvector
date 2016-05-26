#pragma once

#include <cstdio>
#include "model.hpp"
#include "utils.hpp"
#include "config.hpp"

using namespace std;

void read_data(int & D, int & W, document * & docs, float ** & f_r, float ** & f_k) {
    FILE * fin;
    fin = fopen(MAIN_DATA, "r");
    fscanf(fin, "%d %d\n", &D, &W);

    docs = new document[D];

    for (int i = 0; i < D; i ++) {
        int r_id, w_cnt;
        fscanf(fin, "%d %d\n", &r_id, &w_cnt);
        docs[i].r_id = r_id;
        docs[i].w_cnt = w_cnt;
        docs[i].w_id = new int[w_cnt];
        docs[i].w_freq = new int[w_cnt];

        for (int j = 0; j < w_cnt; j ++) {
            fscanf(fin, "%d %d\n", &(docs[i].w_id[j]), &(docs[i].w_freq[j]));
        }
    }

    fclose(fin);

    logging("loading data main done");

    fin = fopen(AUTHOR_EMBEDDING, "r");
    f_r = new float * [D];
    float ** temp_r = new float * [D];
    for (int i = 0; i < D; i ++) {
        f_r[i] = new float[model::E_r];
        temp_r[i] = new float[model::E_r];
    }
    for (int i = 0; i < D; i ++) {
        for (int j = 0; j < model::E_r; j ++)
            // fscanf(fin, "%f\n", &f_r[i][j]);
            fscanf(fin, "%f\n", &temp_r[i][j]);
    }
    fclose(fin);
    for (int i = 0; i < D; i ++) {
        int r_id = docs[i].r_id;
        memcpy(f_r[i], temp_r[r_id], sizeof(float) * model::E_r);
    }

    for (int i = 0; i < D; i ++) delete [] temp_r[i];
    delete [] temp_r;

    logging("loading researcher done");

    fin = fopen(KEYWORD_EMBEDDING, "r");
    f_k = new float*[W];
    for (int i = 0; i < W; i ++) {
        f_k[i] = new float[model::E_k];
    }
    for (int i = 0; i < W; i ++) {
        for (int j = 0; j < model::E_k; j ++)
            fscanf(fin, "%f\n", &f_k[i][j]);
    }
    fclose(fin);

    logging("loading keyword done");
}