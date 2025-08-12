all: advection1D advection2D

advection1D: 
	nvc++  ../src/advection1D.cu -o advection1D 

advection2D: 
	nvc++  ../src/advection2D.cu -o advection2D 

clean:
	rm -rf *.o advection1D advection2D 
