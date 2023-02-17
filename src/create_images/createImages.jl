using Images

# Digital Image Processing - Creating Images
# Author: Hugo Tallys Martins Oliveira

function normalize(M)
    return (M .- minimum(M)) ./ (maximum(M) - minimum(M))
end

range = -5:0.01:5
R(θ) = [cos(θ) sin(θ); -sin(θ) cos(θ)]

s = 2. # scale by a factor of 2.
parab = [ (s*x)^2 + y^2 for x=range, y=range ]
parab = normalize(parab)
par_img = Gray.(parab)

sine_rot = [ 
	begin
		(x, y) = R(π/4) * [x; y] # rotating by pi/4 radians counter clockwise
		sin.(x)
	end
	for x=range, y=range
]

sine_rot = normalize(sine_rot) # normalizing
sine_img = Gray.(sine_rot)

gauss = [ exp(-0.5*(x^2+y^2)) for x=range, y=range ]
gauss_img = Gray.(gauss) # no need to normalize

θ = π / 4
μx, μy = -1.0, 1.0
σx, σy = 1.25, .75

gauss_ = [
	begin
		(x, y) = R(θ) * [x; y]
		exp( -(x - μx)^2 / (2*σx^2) -(y - μy)^2 / (2*σy^2) )
	end
	for x=range, y=range
]

gauss_img_ = Gray.(gauss_)

path = mkpath("imgs/createImages")

save(joinpath(path, "parabola.png"), par_img)
save(joinpath(path, "rotatedSine.png"), sine_img)
save(joinpath(path, "gauss.png"), gauss_img)
save(joinpath(path, "rotatedGauss.png"), gauss_img_)
