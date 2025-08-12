#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include <sys/time.h>
#include <csignal>

using namespace std;

/*transform 2D arr(size NxM) to 1D arr(size N*M) and return pointer arr**/
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
void write(T ** arr,string filename,int nt,int nx,bool binary){
    ofstream out;
    if(binary){
        string name="../TextFiles/"+filename+".bin";
        out.open(name,ios::out|ios::binary);
        if (out.is_open()) {
            out.write(reinterpret_cast<const char*>(&nt), sizeof(int));
            out.write(reinterpret_cast<const char*>(&nx), sizeof(int));
            for (int k = 0; k <nt; k++){
                for (int j = 0; j < nx; j++){
                    out.write(reinterpret_cast<const char*>(&arr[k][j]), sizeof(T));

                }
            }
        }
        else {cout<<"Error while writing"<<endl;}
        out.close();
    }
    else{
        string name="../TextFiles/"+filename+".txt";
        out.open(name,ios::out);
        if (out.is_open()) {
            out << setprecision(20) << nt << " " << setprecision(20) << nx << endl;
            for (int k = 0; k <nt; k++){
                for (int j = 0; j < nx; j++){
                    //if(j%8==0){out << endl;}
                    out << setprecision(20) << arr[k][j] << endl;

                }
                //out << "\n-----------------------------------------------\n" << endl;
                //cout<<k<<endl;
            }
        }
        else {cout<<"Error while writing"<<endl;}
        out.close();
    }

}

/*
 write to arr 2D gausian(mean_x = mx, mean_y = my, dispertion x = sigx^2, dispertion y = sigy^2)

grid is square with size N, length L and start point (x0,y0)
*/
template<typename T> 
void gaussian2D(T ** arr,int N,T x0=0.0f, T y0=0.0f,T L=1.0f,T sigx=10.0f,T sigy=10.0f,T mx=0.5f,T my=0.5f){
    T dx=(L-x0)/(N-1);
    T dy=(L-y0)/(N-1);
    T Dx=pow(sigx,2);
    T Dy=pow(sigy,2);
    for (int i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++){
            arr[i][j] = exp(-(Dx*pow(x0+j*dx-mx,2)+Dy*pow(y0+i*dy-my,2)));
        }        
    }
}


/*calculate initial condition like
    |0|0|1|1|
    |0|0|1|1|
    |0|0|1|1|
    |0|0|1|1|*/
template<typename T> 
void u02(T ** arr,int N){
    for (int i = 0; i < N; i++){
        for (size_t j = 0; j < N/2; j++){
            arr[i][j] = 0;
        }
        for (size_t j = N/2+1; j < N; j++){
            arr[i][j] = 1;
        }         
    }
}

/*
Host function which compute 2D advection and write it to arr
take intial condition as u0,
time method as time_method (pointer to function) which is kernel function,
projection to x of advection velosity as cx (pointer to function),
projection to y of advection velosity as cy (pointer to function),
size of space grid as nx and length of space grid as L,
size of time grid as nt and length of time grid as tmax,
function writes only every t_remember time iteration to arr
*/
template<typename T>
void advection2D(T* u0,T**arr,
                void(*time_metod)(T* ,T* arr,T dt,int nx,T dx,T dy,float2 sig,float2 m),
                int nx,int nt,int t_remember,float2 sig,float2 m,T tmax=1.0f,T L=1.0f){
    int nxy=nx*nx;
    int nBytes=nxy*sizeof(T);
    T dx=L/(nx-1);
    T dy=L/(nx-1);
    T dt =tmax/(nt-1);
    printf("dt=%f \n",dt);
    T* arr_gpu,*buff; // arrays for compute adjection on gpu

    cudaMalloc(&arr_gpu,nBytes);
    cudaMalloc(&buff,nBytes);
    cudaMemcpy(arr_gpu,u0,nBytes,cudaMemcpyHostToDevice);// set initial condition on gpu
    memcpy(arr[0],u0,nBytes);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
    printf("advection2D<<<(%d,%d), (%d,%d)>>> \n", grid.x, grid.y, block.x, block.y);

    for (size_t i = 1; i < nt; i++){
        time_metod<<<grid,block>>>(buff,arr_gpu,dt,nx,dx,dy,sig,m);
        cudaDeviceSynchronize();    
        cudaMemcpy(arr_gpu,buff,nBytes,cudaMemcpyDeviceToDevice);
        if(i%t_remember==0){
            // to reduse size of array we allocate in memory 
            // remember only every t_remember time iteration 
            cudaMemcpy(arr[i/t_remember],buff,nBytes,cudaMemcpyDeviceToHost);
            //printf("%f \n",i*dt);
            
        }
        //printf("\n");
    }
    cudaFree(buff);
    cudaFree(arr_gpu);
}



/*
take point as xy,   sqrt of dispersion as sig,  mean as m 
return vector (type float2) of advection velosity based on gaussian 
*/
__device__ float2 tornado(float2 xy ,float2 sig,float2 m){
    float R=sqrt(pow(xy.x-m.x,2)+pow(xy.y-m.y,2));
    float a=exp(-(R)/sig.x/sig.y)*0.01f;
    float buff=xy.x;
    xy.x=-(xy.y-m.y)/(R+1e-10)*a;
    xy.y=(buff-m.x)/(R+1e-10)*a;    
    //xy.x=1;xy.y=1;
    return xy;
}

__global__ void tornado2(float2* arr, float dx, float dy,int nx,int ny,float2 m,float2 sig,float x0,float y0){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < nx){
        unsigned int idx = iy*nx + ix;
        float2 xy;
        xy.x=ix*dx+x0;
        xy.y=iy*dy+y0;
        float R=sqrt(pow(xy.x-m.x,2)+pow(xy.y-m.y,2));
        float a=exp(-(R)*sig.x*sig.y);
        float buff=xy.x;
        xy.x=-(xy.y-m.y)/(R+1e-10)*a;
        xy.y=(buff-m.x)/(R+1e-10)*a; 
        arr[idx]=xy;
    }
}   
/*
void exlicitEulerStep(float* next,float * arr,
                                void(*space_metod)(),
                                float dx,float dy,float nx,float ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < nx){
        float dxdt=dt/dx;
        unsigned int idx = iy*nx + ix;
        float2 c;c.x=ix*dx;c.y=iy*dy;
        //float2 m_;m_.x=0.5f;m_.y=0.5f;
        c=tornado(c,sig,m);
        float2 w_plus,w_minus;
        w_minus.x=(1+signbit(c.x))/2; w_minus.y=(1+signbit(c.y))/2;
        w_plus.x=1-w_minus.x;w_plus.y=1-w_minus.y;


    }  
}
*/
/*kernel function for compute time iteration
input:
arr - previous time iteration
dt - time step
dx,dy - space steps
sig,m - params for gausian velosity field
output:
next - curent time iteration
*/
template<typename T>
__global__ void explicitEulerStep2D(T* next,T* arr,T dt,int nx,T dx,T dy,float2 sig,float2 m){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < nx){
        float dxdt=dt/dx;
        unsigned int idx = iy*nx + ix;
        float2 c;c.x=ix*dx;c.y=iy*dy;
        //float2 m_;m_.x=0.5f;m_.y=0.5f;
        c=tornado(c,sig,m);
        float2 w_plus,w_minus;
        w_minus.x=(1+signbit(c.x))/2; w_minus.y=(1+signbit(c.y))/2;
        w_plus.x=1-w_minus.x;w_plus.y=1-w_minus.y;

        /*
        split grid into 9 pieces to use left and right diff scheme
        |6|2|7|
        |3|1|4|
        |8|5|9|
        */
        if(ix>0&&iy>0&&ix!=nx&&iy!=nx){// 1
            next[idx]=arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx-1])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[idx-nx])
            -w_minus.x*c.x*dxdt*(arr[idx+1]-arr[idx])
            -w_minus.y*c.y*dxdt*(arr[idx+nx]-arr[idx]);
        }
        else if(iy==0&&ix>0&&ix!=nx){// 2
            //printf("3\t%d = %d-(%d-%d)-(%d-%d)\n",idx,idx,idx,idx-1,idx,nx*(nx-2)-1+idx);
            next[idx] = arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx-1])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[nx*(nx-2)-1+idx])
            -w_minus.x*c.x*dxdt*(arr[idx+1]-arr[idx]);
            -w_minus.y*c.y*dxdt*(arr[idx+nx]-arr[idx]);
        }
        else if(iy>0&&ix==0&&iy!=nx){// 3
            //printf("2\t%d = %d-(%d-%d)-(%d-%d)\n",idx,idx,idx,idx+nx-2,idx,idx-nx);
            next[idx] = arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx+nx-2])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[idx-nx])
            -w_minus.x*c.x*dxdt*(arr[idx+1]-arr[idx]);
            -w_minus.y*c.y*dxdt*(arr[idx+nx]-arr[idx]);
        }
        else if(ix==0&&iy>0&&iy!=nx){// 4
            next[idx]=arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx-1])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[idx-nx])
            -w_minus.x*c.y*dxdt*(arr[idx-nx+2]-arr[idx])
            -w_minus.y*c.y*dxdt*(arr[idx+nx]-arr[idx]);
        }
        else if(iy==nx&&ix>0&&ix!=nx){// 5
            next[idx]=arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx-1])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[idx-nx])
            -w_minus.x*c.x*dxdt*(arr[idx+1]-arr[idx])
            -w_minus.y*c.y*dxdt*(arr[nx+ix]-arr[idx]);
        }
        else if(ix==0&&iy==0) {// 6
            //printf("4\t%d = %d-(%d-%d)-(%d-%d)\n",idx,idx,idx,idx+nx-2,idx,nx*(nx-2));
            next[idx] = arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx+nx-2])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[nx*(nx-2)])
            -w_minus.x*c.x*dxdt*(arr[idx+1]-arr[idx])
            -w_minus.y*c.y*dxdt*(arr[idx+nx]-arr[idx]);
        }
        else if(ix==nx&&iy==nx){//7
            next[idx] = arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx-1])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[nx*(nx-1)-1])
            -w_minus.x*c.x*dxdt*(arr[1]-arr[idx])
            -w_minus.y*c.y*dxdt*(arr[idx+nx]-arr[idx]);
        }
        else if(ix==0&&iy==nx){// 8
            next[idx]=arr[idx]
            -w_plus.x*c.x*dxdt*(arr[nx*nx-2]-arr[idx])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[idx-nx])
            -w_minus.x*c.x*dxdt*(arr[idx+1]-arr[idx])
            -w_minus.y*c.y*dxdt*(arr[nx]-arr[idx]);
        }
        else{ // 9
            next[idx]=arr[idx]
            -w_plus.x*c.x*dxdt*(arr[idx]-arr[idx-1])
            -w_plus.y*c.y*dxdt*(arr[idx]-arr[idx-nx])
            -w_minus.x*c.y*dxdt*(arr[idx-nx+2]-arr[idx]) 
            -w_minus.y*c.y*dxdt*(arr[nx+ix]-arr[idx]);
        }   
    }   
}

// host function for velosity field
float2 torn(float2 xy ,float2 sig,float2 m){
    
    float R=sqrt(pow(xy.x-m.x,2)+pow(xy.y-m.y,2));
    float a=exp(-(R)*sig.x*sig.y);
    float buff=xy.x;
    xy.x=-(xy.y-m.y)/(R+1e-10)*a;
    xy.y=(buff-m.x)/(R+1e-10)*a;    
    return xy;
}

int main(){
    float dt=0.01;
    int nx=64;
    float T=64.0;
    int nt=T/dt+1;
    printf("\nnt = %d\n",nt);

    // every t_remember time iteration is writing to file
    // to reduse memory usage
    int t_remember=log2(nt);
    int size_time=nt/t_remember;

    // params for gausian velosity field
    float2 sig;sig.x=2;sig.y=2; 
    float2 m;m.x=0.6;m.y=0.6;

    // arrays for initial condition
    float ** u0_=new float*[nx];
    for (size_t i = 0; i < nx; i++){u0_[i]=new float[nx];}
    float*u0=new float[nx];

    // array for results
    float ** arr=new float*[size_time+1];
    for (size_t i = 0; i < size_time; i++){arr[i]=new float[nx*nx];}

    // followings lines is for velosity field
    float ** vx=new float*[nx];
    for (size_t i = 0; i < nx; i++){vx[i]=new float[nx];}
    float ** vy=new float*[nx];
    for (size_t i = 0; i < nx; i++){vy[i]=new float[nx];}
    
    float2 xy;
    float dx=1.0f/(nx-1);
    for (size_t i = 0; i < nx; i++){
        for (size_t j = 0; j < nx; j++){
            xy.x=j*dx;xy.y=i*dx;
            xy=torn(xy,sig,m);
            vx[i][j]=xy.x; vy[i][j]=xy.y;
        }    
    }
    float ** v=new float*[2];
    for (size_t i = 0; i < 2; i++){v[i]=new float[nx*nx];}
    float* vx_=new float[nx*nx];
    float* vy_=new float[nx*nx];
    vx_=arr2Dto1D(vx,nx,nx);
    vy_=arr2Dto1D(vy,nx,nx);
    v[0]=vx_;
    v[1]=vy_;
    write(v,"advection2D_velosity_field",2,nx*nx,1);
    // end code for velosity field

    /* calculating and writing solution for initial condition like
    |0|0|1|1|
    |0|0|1|1|
    |0|0|1|1|
    |0|0|1|1|
    */
    u02(u0_,nx);    
    u0=arr2Dto1D(u0_,nx,nx);
    advection2D(u0,arr,explicitEulerStep2D,nx,nt,t_remember,sig,m,T);
    write(arr,"advection2D_hulfzero",size_time,nx*nx,1);

    // calculating andd writing solution for gausian initial condition 
    gaussian2D(u0_,nx,0.0f,0.0f,1.0f,5.0f,5.0f);
    u0=arr2Dto1D(u0_,nx,nx);
    advection2D(u0,arr,explicitEulerStep2D,nx,nt,t_remember,sig,m,T);
    write(arr,"advection2D_gausian",size_time,nx*nx,1);
    

    // free memory
    for (size_t i = 0; i < size_time; i++){delete[] arr[i];}
    for (size_t i = 0; i < nx; i++){delete[] u0_[i];delete []vx[i];delete[] vy[i];}
    delete[] arr; delete[] u0;delete u0_;delete[]vx;delete vy;
    delete[]v[0];delete[] v[1];
    return 0;
}

