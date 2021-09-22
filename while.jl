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


while true

  global i, sqerr, sqerr_p 

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


