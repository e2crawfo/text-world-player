require 'utils'
require 'nn'
require 'rnn'
require 'nngraph'

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.objects    = args.objects
    self.n_objects  = #self.objects
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network

    -- check whether there is a network file
    -- if not (type(self.network) == 'string') then
    --     error("The type of the network provided in NeuralQLearner" ..
    --           " is not a string!")
    -- end

    -- Don't do this anymore, since we're passing in the network that we want to use
    -- local err, msg = pcall(require, self.network)
    -- if not err then
    --     print('Preloading network file:', self.network)
    --     -- try to load saved agent
    --     local err_msg, exp = pcall(torch.load, self.network)
    --     if not err_msg then
    --         error("Could not find network file ")
    --     end
    --     if self.best and exp.best_model then
    --         self.network = exp.best_model
    --     else
    --         self.network = exp.model
    --     end
    -- else
    --     -- Load one of the files in the `embeddings` dir.
    --     print('Creating Agent Network from ' .. self.network)
    --     self.network = msg
    --     self.network = self:network()
    -- end

    -- TODO: Do something smarter here.
    -- if self.gpu == 1 then
    --     self.network:cuda()
    -- else
    --     self.network:float()
    -- end

    -- if self.gpu == 1 then
    --     self.network:cuda()
    --     self.tensor_type = torch.CudaTensor
    -- else
    --     self.network:float()
    --     self.tensor_type = torch.FloatTensor
    -- end
    --
    self.tensor_type = torch.FloatTensor

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Float or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions, numObjects = self.n_objects,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.lastObject = nil
    self.lastAvailableObjects = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    -- self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta, o, available_objects
    local q, q2, q2_max

    s = args.s
    a = args.a
    o = args.o
    r = args.r
    s2 = args.s2
    term = args.term
    available_objects = args.available_objects -- this is for s2

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():typeAs(self.tensor_type()):mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute {max_a Q(s_2, a), max_o Q(s_2, o)}.
    q2_max = target_q_net:forward(s2)

    q2_max_action = q2_max[{{}, {1, self.n_actions}}]:max(2)

    q2_max_object = q2_max[{{}, {self.n_actions+1, self.n_actions+self.n_objects}}]
    q2_max_object:cmul(available_objects)
    q2_max_object[q2_max_object:eq(0)] = -1e30
    q2_max_object = q2_max_object:max(2)

    q2_max = (q2_max_action + q2_max_object) / 2 --take avg. of action and object

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    local target = q2:clone():add(r)

    -- q = Q(s,a)
    local q_all = self.network:forward(s)

    a = a:reshape(a:nElement(), 1)
    o = o:reshape(o:nElement(), 1):add(self.n_actions)

    -- For each batch element, store a vector that is all 0's, except
    -- has the TD-error values for the action/object pair used in the batch-element.
    local err = torch.zeros(target:size(1), self.n_actions + self.n_objects)
    err:scatter(2, a, target - q_all:gather(2, a))
    err:scatter(2, o, target - q_all:gather(2, o))

    target = target:repeatTensor(1, self.n_actions + self.n_objects)

    if self.clip_delta then
        err[err:ge(self.clip_delta)] = self.clip_delta
        err[err:le(-self.clip_delta)] = -self.clip_delta
    end

    if self.gpu == 1 then err = err:cuda() end
    return target, err, q2_max
end


function nql:qLearnMinibatch()
    stime = os.time()

    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw

    local priority_ratio = 0.25 -- fraction of samples from 'priority' transitions
    local s, a, o, r, s2, term, available_objects = self.transitions:sample(self.minibatch_size, priority_ratio)

    stime2 = os.time()
    -- makes calls to forward for both the target network and the active network
    -- Error is the error. Its 0 for actions and objects other than the current action/object.
    local target, err, q2_max = self:getQUpdate{s=s, a=a, o=o, r=r, s2=s2,
        term=term, update_qmax=true, available_objects=available_objects}
    etime2 = os.time()
    print("Time for getQUpdate: " .. etime2 - stime2 .. " seconds.")

    -- zero gradients of parameters
    self.dw:zero()

    if torch.type(self.network) == 'MVRNN' and self.network.train_repr then
        stime3 = os.time()
        mask = torch.zeros(s:size(1), self.n_actions + self.n_objects)

        mask:scatter(2, a:view(a:nElement(), 1), 1.0)
        mask:scatter(2, o:add(self.n_actions):view(o:nElement(), 1), 1.0)

        self.network:accGradReprParameters(s, target, mask)
        etime3 = os.time()
        print("Time for Repr Gradient: " .. etime3 - stime3 .. " seconds.")
    end

    -- get new gradient
    stime4 = os.time()
    self.network:backward(s, err)
    etime4 = os.time()
    print("Time for Predictor Gradient: " .. etime4 - stime4 .. " seconds.")

    stime1 = os.time()
    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    --grad normalization
    -- local max_norm = 50
    -- local grad_norm = self.dw:norm()
    -- if grad_norm > max_norm then
    --   local scale_factor = max_norm/grad_norm
    --   self.dw:mul(scale_factor)
    --   if false and grad_norm > 1000 then
    --       print("Scaling down gradients. Norm:", grad_norm)
    --   end
    -- end

    -- use gradients (original)
    -- self.g:mul(0.95):add(0.05, self.dw)
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g2:mul(0.95):add(0.05, self.tmp) -- g2 is g squared
    -- self.tmp:cmul(self.g, self.g)
    -- self.tmp:mul(-1)
    -- self.tmp:add(self.g2)
    -- self.tmp:add(0.01)
    -- self.tmp:sqrt()

    local smoothing_value = 1e-8
    self.tmp:cmul(self.dw, self.dw)
    self.g:mul(0.9):add(0.1, self.tmp)
    self.tmp = torch.sqrt(self.g)
    self.tmp:add(smoothing_value)  --negative learning rate
    etime1 = os.time()
    print("Time for Gradient Update: " .. etime1 - stime1 .. " seconds.")

    --AdaGrad
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g2:cmul(self.g, self.g)
    -- self.g2:add(self.tmp)
    -- self.g = torch.sqrt(self.g2)
    -- self.tmp = self.g:clone()



    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)

    self.w:add(self.deltas)

    -- print(self.network:parameters())
    -- -- print(self.deltas:size(), self.w:size(), self.dw:size())
    -- -- print(self.deltas:eq(0):sum(), self.w:eq(0):sum(), self.dw:eq(0):sum())
    -- EMBEDDING:updateParameters(self.lr)
    -- print("Deltas: ", self.deltas:norm())
    -- print("W:", self.w:norm())
    -- print("Embedding:",EMBEDDING.weight:norm())
    -- print("Embedding grad weight:",EMBEDDING.gradWeight:norm())
    -- print("Embedding2:",EMBEDDING:forward(torch.range(1, #symbols+1)):norm())
    -- assert(EMBEDDING.gradWeight:norm() > 0)
    etime = os.time()

    print("Time for minibatch: " .. etime - stime .. " seconds.")
end


function nql:sample_validation_data()
    local s, a, o, r, s2, term, available_objects = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_o    = o:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
    self.valid_available_objects = available_objects:clone()
end


function nql:compute_validation_statistics()
    local target, err, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, o = self.valid_o, r=self.valid_r, s2=self.valid_s2,
        term=self.valid_term, available_objects=self.valid_available_objects}

    self.v_avg = self.q_max * q2_max:mean()

    self.tderr_avg = 0
    for i=1,err:size(1) do
        self.tderr_avg = self.tderr_avg + (err[i][self.valid_a[i]] + err[i][self.valid_o[i]]) / 2
    end
    self.tderr_avg = self.tderr_avg / err:size(1)
end


function nql:perceive(reward, state, terminal, testing, testing_ep, available_objects, priority)
    -- Preprocess state (will be set to nil if terminal)
    local curState

    --bounding box for rewards
    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    --Store transition s, a, r, s' (only if not testing)
    -- IMP: available_objects is for selecting an action at s'
    if self.lastState and not testing then
        self.transitions:add(
            self.lastState, self.lastAction, self.lastObject, reward,
            self.lastTerminal,
            table_to_binary_tensor(available_objects, self.n_objects))
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent() -- IMP: curState is with history - not same as 'state'

    -- Select action
    local actionIndex = 1
    local objectIndex = 1
    if not terminal then
        actionIndex, objectIndex, q_func = self:eGreedy(curState, testing_ep, available_objects)
        -- actionIndex, objectIndex, q_func = self:sample_action(curState, testing_ep, available_objects)
    end

    self.transitions:add_recent_action({actionIndex, objectIndex}) --not used

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastObject = objectIndex
    self.lastTerminal = terminal
    self.lastAvailableObjects = available_objects

    if self.target_q and self.numSteps % self.target_q == 1 then
        if torch.type(self.network) == 'MVRNN' or torch.type(self.network) == 'RecNN' then
            self.target_network = self.network:clone_all()
        else
            self.target_network = self.network:clone()
        end

        collectgarbage()
    end

    if not terminal then
        return actionIndex, objectIndex, q_func
    else
        return 0, 0
    end
end


function nql:eGreedy(state, testing_ep, available_objects)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    print("Using Epsilon: " .. self.ep)

    -- Epsilon greedy
    if torch.uniform() < self.ep then
        if not available_objects then
            return torch.random(1, self.n_actions), torch.random(1, self.n_objects)
        else
            return torch.random(1, self.n_actions), available_objects[math.random(#available_objects)]
        end
    else
        return self:greedy(state, available_objects)
    end
end


-- Evaluate all actions (with random tie-breaking)
function nql:getBestRandom(q, N, available_objects)
    if available_objects then
        local maxq = q[available_objects[1]]
        local besta = {available_objects[1]}
        for i=2, #available_objects do
            a = available_objects[i]
            if q[a] > maxq then
                besta = { a }
                maxq = q[a]
            elseif q[a] == maxq then
                besta[#besta+1] = a
            end
        end
        local r = torch.random(1, #besta)
        return besta[r], maxq
    end

    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, N do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end

    local r = torch.random(1, #besta)

    return besta[r], maxq
end


function nql:greedy(state, available_objects)
    -- Doesn't work in batches
    if self.gpu == 1 then
        state = state:cuda()
    end

    q = self.network:forward(state)
    q = q:float():squeeze()

    local best = {}
    local maxq = {}

    local action_q = q[{{1, self.n_actions}}]
    local object_q = q[{{self.n_actions+1, q:nElement()}}]

    best[1], maxq[1] = self:getBestRandom(action_q, self.n_actions)
    best[2], maxq[2] = self:getBestRandom(object_q, self.n_objects, available_objects)

    self.lastAction = best[1]
    self.lastObject = best[2]
    self.bestq = (maxq[1] + maxq[2])/2

    if available_objects then
        q = q:clone()
        q[{{self.n_actions+1, q:nElement()}}]:cmul(available_objects)
    end

    return best[1], best[2], q
end

function nql:_loadNet()
    local net = self.network
    if self.gpu == 1 then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print("Weight mean:\n" .. torch.mean(torch.abs(self.w)))
    print("Weight max:\n" .. torch.abs(self.w):max())

    print("Weight grad mean:\n" .. torch.mean(torch.abs(self.dw)))
    print("Weight grad max:\n" .. torch.abs(self.dw):max())
end
