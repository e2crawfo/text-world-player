require 'nn'
require 'nnx'
require 'nngraph'

--require 'cunn'
-- IMP if args is not passed, it takes from global 'args'

local function make_rnn(state_dim, hist_len, gpu)
    rho = state_dim --number of backprop steps
    n_hid = EMBEDDING.weight:size(2)

    r = nn.Recurrent(
       n_hid, EMBEDDING,
       nn.Linear(n_hid, n_hid),
       nn.Rectifier(),
       rho
    )

    rnn = nn.Sequential()

    rnn_seq = nn.Sequential()
    rnn_seq:add(nn.Sequencer(r))

    -- e2crawfo: It used to do this instead of mean pooling.
    -- rnn_seq:add(nn.SelectTable(state_dim))
    -- rnn_seq:add(nn.Linear(n_hid, n_hid))
    -- rnn_seq:add(nn.Rectifier())

    -- Mean pooling
    rnn_seq:add(nn.CAddTable())
    rnn_seq:add(nn.Linear(n_hid, n_hid))
    rnn_seq:add(nn.Rectifier())

    parallel_flows = nn.ParallelTable()
    for f=1, hist_len do
        if f > 1 then
            parallel_flows:add(rnn_seq:clone("weight","bias", "gradWeight", "gradBias"))
        else
            parallel_flows:add(rnn_seq)
        end
    end

    rnn:add(parallel_flows)
    rnn:add(nn.JoinTable(2))

    if gpu == 1 then
        rnn:cuda()
    end

    return rnn, n_hid * hist_len
end

rnn = {
    make_rnn=make_rnn
}

return rnn
