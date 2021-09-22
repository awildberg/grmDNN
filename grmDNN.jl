
using Images, Plots, Distributions, LinearAlgebra, JLD

cd(homedir())
cd(".\\Documents\\Julia")
#cd("/mnt/c/Users/User/Documents/Julia/")

#file_no = Array{Int}( floor.( (rand(1000) *59999) .+1 ) )
file_no = sample( 1:60000, 60000, replace=false );

s = readdir("./mnist_png/training/");

dat = hcat( reshape.( Array{Float32,2}.( Images.load.( string.("./mnist_png/training/", s[file_no]) ) ), :,1 )... );

target_no = ( parse.( Int32, replace.( s[file_no], r"_.*" => "" ) ) ) .+1;
target = Array{Bool}( I, 10, 10 );


s_t = readdir("./mnist_png/testing/");

dat_t = hcat( reshape.( Array{Float32,2}.( Images.load.( string.("./mnist_png/testing/", s_t) ) ), :,1 )... );
target_t_no = ( parse.( Int32, replace.( s_t, r"_.*" => "" ) ) );


nn = Int32[100,100,10]

w = [ Array{Float32}( rand( Distributions.Normal( 0,0.5) ,784, nn[1]) ) ] # Array{Float32,2}[repeat( Float32[1e-6], 784, nn[1] )] # initialize layer
append!(w, [ Array{Float32}( rand( Distributions.Normal( 0,0.3), nn[1], nn[2]) ) ])
append!(w, [ Array{Float32}( rand( Distributions.Normal( 0,0.2), nn[2], nn[3]) ) ])



percent_zero = Int8[40, 30, 0]

w[1][ sample( 1:length(w[1]), Int32.(floor.(length(w[1]) * (percent_zero[1]/100))), replace=false ) ] .= 0
w[2][ sample( 1:length(w[2]), Int32.(floor.(length(w[2]) * (percent_zero[2]/100))), replace=false ) ] .= 0
w[3][ sample( 1:length(w[3]), Int32.(floor.(length(w[3]) * (percent_zero[3]/100))), replace=false ) ] .= 0

### JLD.@load "mnist.jld"

dat_ = dat'
###
l_1 = dat_ * w[1] # size 1000, 100
l_1_norm = l_1 ./ maximum( abs.(l_1),  dims=1 ) # maxnorm size 1000, 100
l_2 = l_1_norm * w[2] # size 1000, 40
l_2_norm = l_2 ./ maximum( abs.(l_2),  dims=1 ) # exp.( Array{Float64}(l_2) ) ./ sum( exp.( Array{Float64}(l_2) ), dims=2 ) # softmax size 1000, 10
l_out = l_2_norm * w[3] # size 1000, 10
l_out_norm = exp.( Array{Float64}(l_out) ) ./ sum( exp.( Array{Float64}(l_out) ), dims=2 ) # softmax size 1000, 10
sqerr_p = sum( sum( ( l_out_norm - target[ target_no[1:end], :] ).^2, dims=1 ) )
###




change = Float32[0,0,0]
change = [zeros( Float32, size(w[1]) )]
append!( change, [zeros( Float32, size(w[2]) )] )
append!( change, [zeros( Float32, size(w[3]) )] )

mutate_w = Int32[0,0,0]

tmp = Float32[0,0,0]

first = 0 # sample(3, 1)

probe = Float32[0,0,0]

p = Int8[1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,3]
#p = Int8[2,2,2]



function pred( data_vect, w_1, w_2, w_3 )

	l_1 = transpose( data_vect ) * w_1
	l_norma = l_1 / maximum( abs.(l_1)) # maxnorm
	l_2 = l_norma * w_2
	l_norma = l_2 / maximum( abs.(l_2)) # maxnorm
	l_3 = l_norma * w_3
	return findmax( exp.(Array{Float64}(l_3)) / sum( exp.(Array{Float64}(l_3)) ) )[2][2] -1
	#return exp.(Array{Float64}(l_3)) / sum( exp.(Array{Float64}(l_3)) )

end

#sqerr_p = 10000

sqerr = 0

i = 0

while true

  if(mod(i, 500) == 0 && sqerr == sqerr_p)
    xx = Bool[]
    for i in 1:10000 push!(xx, (target_t_no[i] == pred( dat_t[:,i], w[1], w[2], w[3] )) ) end
    println( length(xx[ xx.==true ]) / 10000, " ", maximum.([w[1],w[2],w[3]]), minimum.([w[1],w[2],w[3]]) )
    println( length( change[1][ change[1] .== 0 ] ) / length( change[1] ), " ", length( change[2][ change[2] .== 0 ] ) / length( change[2] ), " ", length( change[3][ change[3] .== 0 ] ) / length( change[3] ) )
  end

  probe[1] = Float32.(rand( Distributions.Normal(0, 0.4), 1) )[1] #Float32.( rand( Distributions.Chisq(2), 1 )/8 )[1] # probe <- runif(1)
  probe[2] = Float32.(rand( Distributions.Normal(0, 0.3), 1) )[1] # probe <- runif(1)
  probe[3] = Float32.(rand( Distributions.Normal(0, 0.1), 1) )[1] # probe <- runif(1)

  first = p[sample(1:16, 1)][1]

  mutate_w[first] = sample( 1:length(w[first]), 1, replace= false)[1]

  if w[first][ mutate_w[first] ] == false continue end

  tmp[first] = w[first][ mutate_w[first] ]
  w[first][ mutate_w[first] ] = w[1][ mutate_w[first] ] + probe[first]

    ###
    l_1 = dat_ * w[1] # size 1000, 100
    l_1_norm = l_1 ./ maximum( abs.(l_1),  dims=1 ) # maxnorm size 1000, 100
    l_2 = l_1_norm * w[2] # size 1000, 40
    l_2_norm = l_2 ./ maximum( abs.(l_2),  dims=1 ) # exp.( Array{Float64}(l_2) ) ./ sum( exp.( Array{Float64}(l_2) ), dims=2 ) # softmax size 1000, 10
    l_out = l_2_norm * w[3] # size 1000, 10
    l_out_norm = exp.( Array{Float64}(l_out) ) ./ sum( exp.( Array{Float64}(l_out) ), dims=2 ) # softmax size 1000, 10
    sqerr = sum( sum( ( l_out_norm - target[ target_no[1:end], :] ).^2, dims=1 ) )
    ###
	
  if sqerr < sqerr_p
    i += 1
    #println( w[first][ mutate_w[first] ], " ", probe[first] )
    if(mod(i, 10)==0) println("+ ", sqerr, " | ", sqerr_p, "\t", first, " ", w[first][ mutate_w[first] ], " ", probe[first], "\t",i ) end
      change[first][ mutate_w[first] ] += ( sqerr_p - sqerr )
      sqerr_p = sqerr

  else
    w[first][ mutate_w[first] ] = tmp[first]
    w[first][ mutate_w[first] ] -= probe[first]

    ###
    l_1 = dat_ * w[1] # size 1000, 100
    l_1_norm = l_1 ./ maximum( abs.(l_1),  dims=1 ) # maxnorm size 1000, 100
    l_2 = l_1_norm * w[2] # size 1000, 40
    l_2_norm = l_2 ./ maximum( abs.(l_2),  dims=1 ) # exp.( Array{Float64}(l_2) ) ./ sum( exp.( Array{Float64}(l_2) ), dims=2 ) # softmax size 1000, 10
    l_out = l_2_norm * w[3] # size 1000, 10
    l_out_norm = exp.( Array{Float64}(l_out) ) ./ sum( exp.( Array{Float64}(l_out) ), dims=2 ) # softmax size 1000, 10
    sqerr = sum( sum( ( l_out_norm - target[ target_no[1:end], :] ).^2, dims=1 ) )
    ###
    if sqerr < sqerr_p
      i += 1
      #println( w[first][ mutate_w[first] ], " ", probe[first] )
      if(mod(i, 10)==0) println("- ", sqerr, " | ", sqerr_p, "\t", first, " ", w[first][ mutate_w[first] ], " ", probe[first], "\t",i ) end
      change[first][ mutate_w[first] ] += ( sqerr_p - sqerr )
      sqerr_p = sqerr

      else
        #println(sqerr - sqerr_p, " down... ",first)
        w[first][ mutate_w[first] ] = tmp[first]

    end

  end

end




function pred( data_vect, w_1, w_2, w_3 )

	l_1 = transpose( data_vect ) * w_1
	l_norma = l_1 / maximum( abs.(l_1)) # maxnorm
	l_2 = l_norma * w_2
	l_norma = l_2 / maximum( abs.(l_2)) # maxnorm
	l_3 = l_norma * w_3
	return findmax( exp.(Array{Float64}(l_3)) / sum( exp.(Array{Float64}(l_3)) ) )[2][2] -1
	#return exp.(Array{Float64}(l_3)) / sum( exp.(Array{Float64}(l_3)) )

end

pred( dat[:,89], w[1], w[2], w[3] )

xx = Bool[]
for i in 1:10000 push!(xx, (target_t_no[i] == pred( dat_t[:,i], w[1], w[2], w[3] )) ) end
length(xx[ xx.==true ]) / 10000

JLD.@save "mnist.jld" w change

JLD.@load "mnist.jld"





function center( arr )
  ( ( abs( minimum( arr ) ) - maximum( arr ) ) / ( abs( minimum( arr ) ) + abs( maximum( arr ) ) ) * .5 ) + .5
end
heatmap( w[1]', c=cgrad( :bwr, [ center(w[1]) ] ) )


