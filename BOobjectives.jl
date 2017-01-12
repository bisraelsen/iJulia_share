using Distributions
function gramacy(num_samp::Int; opt_range=(0.5,2.5), add_noise=true, noise=Normal(0,1.5))
    dist = Uniform(opt_range[1],opt_range[2])
    x = rand(dist,num_samp)
    y = gramacy(x,opt_range,add_noise=add_noise)
    return x,y,opt_range
end

function gramacy(x::Array, opt_range=(0.5,2.5);add_noise=false, noise=Normal(0,0.5))
    if add_noise
        y = (sin(10*pi.*x)./2.*x + (x-1).^4) + rand(noise,length(x))
    else
        y = (sin(10*pi.*x)./2.*x + (x-1).^4)
    end

    return y
end

function forrester(num_samp::Int;opt_range::Tuple=(0.0,1.0),add_noise::Bool=true, noise::Distributions.Distribution=Normal(0,0.5))
    dist = Uniform(opt_range[1],opt_range[2])
    x = rand(dist,num_samp)
    y = forrester(x,opt_range,add_noise=add_noise,noise=noise)
    return x,y,opt_range
end

function forrester(x::Array, opt_range=(0.0,1.0);add_noise=false, noise=Normal(0,0.5))
    if add_noise
        noise_vals =  rand(noise,length(x))
        if size(x) != size(noise_vals)
            println("size is wrong, attempting to correct")
            x = x'
        end
        y = ((6.*x-2).^2.*sin(12.*x-4)) + noise_vals
    else
        y = ((6.*x-2).^2.*sin(12.*x-4))
    end

    return y
end


