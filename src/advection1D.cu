#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include <sys/time.h>
#include <csignal>

using namespace std;

/*transform 2D arr (size NxM) to 1D arr(size N*M) and return pointer arr**/
template<typename T>
T* arr2Dto1D(T ** arr,int N,int M){
    T* arr2=new T[N*M];
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            arr2[i*M+j]=arr[i][j];
    
    return arr2;
}

/*transform 1D arr(size N*M) to 2D arr(size NxM) and return pointer arr***/
template<typename T>
T** arr1Dto2D(T * arr,int N,int M){
    T** arr2=new bool*[N];
    for (int i = 0; i < N; i++){arr2[i] = new bool [M];}

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            arr2[i][j]=arr[i*M+j];
    return arr2;
}

/* arr = gaussian(mean,sig) for x in [x0;L], N is number of elements in arr
default:
mean = 0.5
sig=10
x0=0
L=1*/
template<typename T> 
void gaussian(T * arr,int N,T sig=10.0f,T mean=0.5f,T x0=0.0f,T L=1.0f){
    T dx=(L-x0)/(N-1);
    T D=pow(sig,2);
    for (int i = 0; i < N; i++){
        arr[i] = exp(-D*pow(x0+i*dx-mean,2));
    }
}

/*print 2D arr*/
template<typename T> 
void show2D(T ** arr,int N,int M){
    for (size_t i = 0; i <N; i++){
        std::cout<<std::endl;
        for (size_t j = 0; j < M; j++)
            std::cout<<arr[i][j]<<"\t";        
    }
    std::cout<<std::endl;
}

/*print 1D arr*/
template<typename T> 
void show1D(T * arr,int N){
    for (size_t i = 0; i <N; i++)
        std::cout<<arr[i]<<"\t";
    cout<<endl;
}

template<typename T>
__global__ void explicitEulerStep1D(T* next,T* arr,T dt,int nx,T c,T dx){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    if (ix < nx && iy < nx){
        int N=nx-1;
        unsigned int idx = iy*nx + ix;
        //printf("%d\n",idx);
        if(idx>0){
            next[idx]=arr[idx]-dt*c/dx*(arr[idx]-arr[idx-1]);
        }
        else{
            next[idx] = arr[N]-dt*c/dx*(arr[N]-arr[N-1]); 
        }
    }    

}

template<typename T>
void adjection1D(T* u0,T* res,void(*time_metod)(T*,T*,T,int,T,T),T c=1.0f,int nx =1024,int nt=100,T tmax=1.0f,T L=1.0f){
    int nBytes=nx*sizeof(T);
    T dx=L/(nx-1);
    T dt =tmax/(nt-1);
    printf("CFL = %lf\n",c*dt/dx);
    
    T* arr_gpu,*buff; // arrays for compute adjection on gpu
    cudaMalloc(&arr_gpu,nBytes);
    cudaMalloc(&buff,nBytes);
    cudaMemcpy(arr_gpu,u0,nBytes,cudaMemcpyHostToDevice);// set initial condition on gpu

    int block_x=nx>>int(log2(nx))/2;
    dim3 block(block_x);
    dim3 grid((nx + block.x - 1) / block.x);
    printf("advection1D_gpu<<<(%d,%d), (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);
    for (size_t i = 1; i < nt; i++){
        time_metod<<<grid,block>>>(buff,arr_gpu,dt,nx,c,dx);
        cudaDeviceSynchronize();    
        cudaMemcpy(arr_gpu,buff,nBytes,cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(res,buff,nBytes,cudaMemcpyDeviceToHost);
    cudaFree(buff);
    cudaFree(arr_gpu);
    
}
  
template<typename T>
void write(T ** arr,string filename,int nt,int nx){
    ofstream out;
    
    string name="../TextFiles/"+filename+".bin";
    out.open(name,ios::out|ios::binary);
    if (out.is_open()) {
        out.write(reinterpret_cast<const char*>(&nt), sizeof(int));
        out.write(reinterpret_cast<const char*>(&nx), sizeof(int));
        for (int k = 0; k <nt; k++){
            for (int j = 0; j < nx; j++){
                out.write(reinterpret_cast<const char*>(&arr[k][j]), sizeof(T));

            }
            //out << "\n-----------------------------------------------\n" << endl;
            //cout<<k<<endl;
        }
    }
    else {cout<<"Error while writing"<<endl;}
    out.close();
}

template<typename T>
void write_norm(T * arr,string filename,int n){
    ofstream out;
    string name="../TextFiles/"+filename+".bin";
    out.open(name,ios::out|ios::binary);
    if (out.is_open()) {
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (int k = 0; k <n; k++)
            out.write(reinterpret_cast<const char*>(&arr[k].x), sizeof(T)/2);
        
        for (int k = 0; k <n; k++)
            out.write(reinterpret_cast<const char*>(&arr[k].y), sizeof(T)/2);
        
    }
    else {cout<<"Error while writing"<<endl;}
    out.close();
}

/*calculate l2 norm of error between exatc and numerical*/
template<typename T>
T l_2(T*numerical,T*exact,int n){
    T res=0;
    for (size_t i = 0; i < n; i++){
        res+=pow(numerical[i]-exact[i],2);
    }
    return sqrt(res/n);
}

/*calculate infinum norm of error between exatc and numerical*/
template<typename T>
T l_inf(T*numerical,T*exact,int n){
    T max=0;
    T buff;
    for (size_t i = 0; i < n; i++){
        buff=numerical[i]-exact[i];
        if (buff>max){max=buff;}
    }
    return max;
}

int main(){

    int max_n=18;
    int start=6;
    int N=max_n-start;
    
    float2 * l_2_arr=new float2[N];
    float2 * l_inf_arr=new float2[N];


    for(int i=start;i<max_n;i++){
        int nx=pow(2,i);
        int nt = 2*nx;
        float *u0=new float[nx];
        float* res=new float[nx];
        gaussian(u0,nx);
        adjection1D(u0,res,explicitEulerStep1D,1.0f,nx,nt);
        l_2_arr[i-start].x=nx;
        l_2_arr[i-start].y=l_2(res,u0,nx);
        l_inf_arr[i-start].x=nx;
        l_inf_arr[i-start].y=l_inf(res,u0,nx);
        delete[]u0;
        delete[]res;
    }

    write_norm(l_2_arr,"advection1D_l2norm",N);
    write_norm(l_inf_arr,"advection1D_linfnorm",N);
    //free
    delete[] l_2_arr; delete[] l_inf_arr;
    return 0;
}
