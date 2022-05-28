
{
	THREADS=36

	wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.5-linux-x86_64.tar.gz
	tar zxvf julia-1.6.5-linux-x86_64.tar.gz

	export PATH=$PATH:/home/ec2-user/julia-1.6.5/bin

	# install the required julia packages...
	julia -e 'using Pkg; Pkg.add("HomotopyContinuation")'
	julia -e 'using Pkg; Pkg.add("Distributions")'
	julia -e 'using Pkg; Pkg.add("ArgParse")'
	julia -e 'using Pkg; Pkg.add("OrderedCollections")'
	julia -e 'using Pkg; Pkg.add("JSON")'

	cd ./core

	echo "Starting main.jl with $THREADS threads..."
	julia --threads $THREADS main.jl
	echo "Process completed successfully!"

} &>> stdout.txt
