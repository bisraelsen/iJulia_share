using PyPlot, GaussianProcesses, Debug
include("BOobjectives.jl")

function t1(gp;x=gp.X,y=gp.y,sig=1.0)
    kt = cov(gp.k,x)
    si = sig*eye(similar(kt))

    t1 = (y'*(kt+si)^-1*y)[1]
    condkt = cond(kt)
    condt1 = cond(kt+si)
    return t1, condkt, condt1
end

function t2(gp;x=gp.X,sig=1.0)
    kt = cov(gp.k,x)
    si = sig*eye(similar(kt))
    return log(det(kt+si))
end

function sweep_hyp(x,y,gp,obj_fxn;rs=1,hyp_val="ℓ2",xs_rng=linspace(0.01,1.0,100))
    new_xs = repmat(x,rs,1)'
    new_ys = obj_fxn(new_xs)[:]

    xs_ms, ys_ms, opt_rng = obj_fxn(length(x))

    n = 100
    ll = []
    ll_ms = []
    for i in xs_rng
        if contains(hyp_val,"ℓ")
            gp.k.ℓ2 = i
        else
            gp.k.σ2 = i
        end
        new_gp = GP(new_xs,new_ys,gp.m,gp.k,gp.logNoise)
        gp_ms = GP(xs_ms,ys_ms,gp.m,gp.k,gp.logNoise)
        t1val = -0.5*t1(new_gp)[1]
        t2val = -0.5*t2(new_gp)
        normalization = -length(x)/2*log(2*pi)
        push!(ll,t1val+t2val+normalization)

        t1ms = -0.5*t1(gp_ms)[1]
        t2ms = -0.5*t2(gp_ms)
        push!(ll_ms,t1ms+t2ms+normalization)
    end
    return ll,ll_ms,xs_rng
end
function sweep_noise(x,y,gp,obj_fxn;rs=1,xs_rng=linspace(0.01,1.0,100))
    new_xs = repmat(x,rs,1)'
    new_ys = obj_fxn(new_xs)[:]
    new_gp = GP(new_xs,new_ys,gp.m,gp.k,gp.logNoise)

    xs_ms, ys_ms, opt_rng = obj_fxn(length(x))
    gp_ms = GP(xs_ms,ys_ms,gp.m,gp.k,gp.logNoise)

    n = 100
    ll = []
    ll_ms = []
    for i in xs_rng
        t1val = -0.5*t1(new_gp,sig=i)[1]
        t2val = -0.5*t2(new_gp,sig=i)
        normalization = -length(x)/2*log(2*pi)
        push!(ll,t1val+t2val+normalization)

        t1ms = -0.5*t1(gp_ms,sig=i)[1]
        t2ms = -0.5*t2(gp_ms,sig=i)
        push!(ll_ms,t1ms+t2ms+normalization)
    end
    return ll,ll_ms,xs_rng
end
function condition_exploration()
    # probably broken, I just stuck this here to get it out of the main stream
    x = xs(2)
    t1_ary = []
    t2_ary = []
    condkt_ary = []
    condt1_ary = []
    for i = 2:10
        new_xs = repmat(x,i,1)'
        new_ys = ys(new_xs)[:]
        gp = GP(new_xs,new_ys,mZero,kern,logObsNoise)
        out = t1(gp)
        outt2= t2(gp)
        push!(t1_ary,out[1])
        push!(t2_ary,outt2[1])
        push!(condkt_ary,out[2])
        push!(condt1_ary,out[3])
    end
end

function function_test(obj_fxn,num_samp,acq_noise)
    x,y,opt_range = obj_fxn(num_samp)
    kern = SE(0.0,0.0)
    mZero = MeanZero()
    logObsNoise = 0.0
    gp = GP(x,y,mZero,kern,logObsNoise)

    fig = PyPlot.figure()
    scatter(x,y)

    fig,ax_ary = PyPlot.subplots(3)
    fig[:set_size_inches](6.0,8.0)
    lbl_size = 25
    x_rng_noise = linspace(0.01,5,1000)
    x_rng = linspace(0.01,5,1000)
    for i in [1,5,10]
        ll_vals = sweep_noise(x,y,gp,obj_fxn,rs=i,xs_rng=x_rng_noise)
        ax_ary[1][:set_title](@sprintf("Initial Samples: %d, noise: %0.2f",num_samp,acq_noise))
        ax_ary[1][:plot](ll_vals[3],ll_vals[1],label=@sprintf("RS%d",i),lw=2,ls="dashed")
        ax_ary[1][:plot](ll_vals[3],ll_vals[2],label=@sprintf("MS%d",i))
        ax_ary[1][:set_xlabel](L"\sigma_n^2",fontsize=lbl_size)
        ax_ary[1][:set_ylabel](L"log_lik",fontsize=lbl_size)
    end
    for i in [1,5,10]
        hyp_name = "ℓ2"
        ll_vals = sweep_hyp(x,y,gp,obj_fxn,rs=i,hyp_val=hyp_name,xs_rng=x_rng)
        ax_ary[2][:plot](ll_vals[3],ll_vals[1],label=@sprintf("RS%d",i),lw=2,ls="dashed")
        ax_ary[2][:plot](ll_vals[3],ll_vals[2],label=@sprintf("MS%d",i))
        ax_ary[2][:set_xlabel](L"\ell",fontsize=lbl_size)
        ax_ary[2][:set_ylabel](L"log_lik",fontsize=lbl_size)
    end
    for i in [1,5,10]
        hyp_name = "σ2"
        ll_vals = sweep_hyp(x,y,gp,obj_fxn,rs=i,hyp_val=hyp_name,xs_rng=x_rng)
        ax_ary[3][:plot](ll_vals[3],ll_vals[1],label=@sprintf("RS%d",i),lw=2,ls="dashed")
        ax_ary[3][:plot](ll_vals[3],ll_vals[2],label=@sprintf("MS%d",i))
        ax_ary[3][:set_xlabel](L"\sigma_o^2",fontsize=lbl_size)
        ax_ary[3][:set_ylabel](L"log_lik",fontsize=lbl_size)
    end
    PyPlot.legend()
end

obj_fxn = forrester
num_samp = 25
acq_noise = 5.0
function obj_fxn_opt(x::Int)
    ret_val = obj_fxn(x,add_noise=true,noise=Normal(0.0,acq_noise))
    return ret_val[1], ret_val[2], ret_val[3]
end
function obj_fxn_gp(obj_fxn_opt)
    # make a GP that is based on a real objective function
    num_pts = 500

    (x,y,bnds) = obj_fxn_opt(num_pts)

    kern = SE(0.1,1.0)
    mZero = MeanZero()
    logObsNoise = 2.0

    gp = GP(x,y,mZero,kern,logObsNoise)
    optimize!(gp)
    fig = PyPlot.figure()
    plot(gp)
    return gp
end

obj_fxn_opt(x::Array) = -1.0 * obj_fxn(x,add_noise=true,noise=Normal(0.0,acq_noise))
forrester_gp = obj_fxn_gp(obj_fxn_opt)

function forrester_gp_opt(x::Int)
    ret_val = obj_fxn(x)
    ys = forrester_gp_opt(ret_val[1])

    return ret_val[1], ys, ret_val[3]
end
function forrester_gp_opt(x::Array)
    vals = zeros(length(x))
    for i = 1:length(x)
        vals[i] = rand(forrester_gp,[x[i]],1)[1]
    end
    return vals
end

# function_test(forrester_gp_opt,num_samp,acq_noise)

