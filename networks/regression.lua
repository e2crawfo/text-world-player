require 'nn'
--require 'cunn'
-- IMP if args is not passed, it takes from global 'args'
--
-- n_input = args.hist_len*args.ncols*args.state_dim
-- n_hid = 100
local function make_deep_regressor (n_input, n_hid, n_actions, n_objects, gpu)

    local mlp = nn.Sequential()

    mlp:add(nn.Reshape(n_input))
    mlp:add(nn.Linear(n_input, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())

    mlp:add(nn.Linear(n_hid, n_actions + n_objects))

    if gpu == 1 then
        mlp:cuda()
    end

    return mlp
end

-- n_input = args.hist_len*args.ncols*args.state_dim
local function make_shallow_regressor (n_input, n_actions, n_objects, gpu)

    local mlp = nn.Sequential()

    mlp:add(nn.Reshape(n_input))
    mlp:add(nn.Linear(n_input, n_actions + n_objects))

    if gpu == 1 then
        mlp:cuda()
    end

    return mlp
end

regression = {
    make_deep_regressor=make_deep_regressor,
    make_shallow_regressor=make_shallow_regressor
}

return regression
