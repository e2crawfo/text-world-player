-- agent
require 'torch'
package.path = package.path .. ';networks/?.lua'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-exp_folder', '', 'name of folder where current exp state is being stored')
cmd:option('-text_world_location', '', 'location of text-world folder')

cmd:option('-actrep', 1, 'how many times to repeat action')

cmd:option('-representation', '', 'The type of representation to use. Can be "bow", "bob", "mvrnn", "rnn", or "lstm".')
cmd:option('-regressor', '', 'The type of regressor to use. Can be "deep" or "shallow".')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')

cmd:option('-quest_levels', 1,'# of quests to complete in each run')
cmd:option('-recurrent_dim', 100, 'max dimensionality of recurrent state (stream of symbols)')
cmd:option('-max_steps', 100, 'max steps per episode')

cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-game_num', 1, 'game number (for parallel game servers)')
cmd:option('-tutorial_world', 1, 'play tutorial_world')
cmd:option('-random_test', 0, 'test random policy')
cmd:option('-analyze_test', 0, 'load model and analyze')

cmd:option('-wordvec_file', 'wordvec.eng' , 'Word vector file')
cmd:option('-use_wordvec', 0, 'use word vec')

cmd:text()

local opt = cmd:parse(arg)
print(opt)

assert (opt.representation ~= '', "-representation must be supplied.")

RECURRENT = opt.representation == 'rnn' or opt.representation == 'lstm'
QUEST_LEVELS = opt.quest_levels
MAX_STEPS = opt.max_steps
WORDVEC_FILE = opt.wordvec_file
TUTORIAL_WORLD = (opt.tutorial_world==1)
RANDOM_TEST = (opt.random_test==1)
ANALYZE_TEST = (opt.analyze_test==1)

print("Using Tutorial world?", TUTORIAL_WORLD)

require 'client'
require 'utils'
require 'xlua'
require 'optim'

local framework
if TUTORIAL_WORLD then
    framework = require 'framework_fantasy'
else
    framework = require 'framework'
end
---------------------------------------------------------------

-- e2crawfo: This seems to be causing some weird version of underscore to be imported...
if not dqn then
    dqn = {}
    require 'nn'
    require 'nngraph'
    require 'nnutils'
    require 'NeuralQLearner'
    require 'TransitionTable'
    require 'Rectifier'
    require 'Embedding'

    require 'networks/rnn'
    require 'networks/lstm'
    require 'networks/mvrnn'
    require 'networks/text_to_vector.lua'
    require 'networks/regression.lua'
end

-- Need to do it here so that the earlier requires don't overwrite _
_ = require 'underscore'

-- agent login
local port = 4000 + opt.game_num
print(port)
client_connect(port)
login('root', 'root')

if TUTORIAL_WORLD then
    framework.makeSymbolMapping(opt.text_world_location .. 'evennia/contrib/tutorial_world/build.ev')
else
    framework.makeSymbolMapping(opt.text_world_location .. 'evennia/contrib/text_sims/build.ev')
end

print (opt.agent_params)

state_dim = framework.getStateDim()


-- General setup.
if opt.agent_params then
    opt.agent_params = str_to_table(opt.agent_params)
    opt.agent_params.gpu       = opt.gpu
    opt.agent_params.best      = opt.best
    opt.agent_params.verbose   = opt.verbose

    opt.agent_params.actions = framework.getActions()
    opt.agent_params.objects = framework.getObjects()

    opt.agent_params.state_dim = state_dim
end


wv_dim = 20
random_vec_func = (
    function (size)
        return torch.rand(size)*0.02-0.01
    end
)
wv_init = opt.use_wordvec == 1 and readWordVec(WORDVEC_FILE) or nil
wv_func = nil

if wv_init then

    function make_wv_func (wv_init)
        local function f (size, word)
            return torch.Tensor(size):copy(wv_init[word])
        end

        return f
    end

    wv_func = make_wv_func(wv_init)
else
    wv_func = random_vec_func
end


-- Make the representation
rep = opt.representation
rep_network = nil
rep_dim = nil

if rep == "bow" then
    print ("Using " .. rep)
    rep_network, rep_dim = text_to_vector.make_bow(symbols, symbol_mapping)
elseif rep == "bob" then
    print ("Using " .. rep)
    rep_network, rep_dim = text_to_vector.make_bob(symbols, symbol_mapping)
elseif rep == "mvrnn" then
    print ("Using " .. rep)
    r = 3
    nl_class = nn.Tanh

    rep_network, rep_dim = mvrnn.make_mvrnn(
         wv_dim, r, nl_class, mvrnn.CTS, mvrnn.MEAN, random_vec_func, wv_func)

elseif rep == "lstm" or "rnn" then

    recurrent_dim = opt.recurrent_dim
    ol_network, ol_dim = text_to_vector.make_ordered_list(symbols, symbol_mapping, recurrent_dim)

    -- Need to set this global variable before creating LSTM or RNN
    EMBEDDING = Embedding(#symbols+1, wv_dim)
    EMBEDDING:setWordVecs(symbols, wv_func)

    if rep == "lstm" then
        print ("Using " .. rep)
        rep_network, rep_dim = lstm.make_lstm(opt.agent_params.hist_len, opt.gpu)
    else
        print ("Using " .. rep)
        rep_network, rep_dim = rnn.make_rnn(
            recurrent_dim, opt.agent_params.hist_len, opt.gpu)
    end

    rep_network = (
        nn.Sequential()
        :add(ol_network)
        :add(rep_network))
else
    error("Invalid representation `" .. rep .. "` supplied.")
end


-- Make the regressor
regressor = opt.regressor
regressor_network = nil

n_actions = #(framework.getActions())
n_objects = #(framework.getObjects())

print(n_actions, n_objects, rep_dim, opt.gpu)

if regressor == "shallow" then
    print ("Using " .. regressor)
    regressor_network = regression.make_shallow_regressor(rep_dim, n_actions, n_objects, opt.gpu)
elseif regressor == "deep" then
    print ("Using " .. regressor)
    n_hid = 100
    regressor_network = regression.make_deep_regressor(rep_dim, n_hid, n_actions, n_objects, opt.gpu)
else
    error("Invalid regressor `" .. regressor .. "` supplied.")
end


-- Put the pieces together
opt.agent_params.network = (
    nn.Sequential()
    :add(rep_network)
    :add(regressor_network)
)


-- Initialize the agent
local agent = dqn.NeuralQLearner(opt.agent_params)

-- Override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local bestq_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

local state, reward, terminal, available_objects = framework.newGame()
local priority = false

print("Started RL based training ...")
local pos_reward_cnt = 0
local quest1_reward_cnt, quest2_reward_cnt, quest3_reward_cnt


print('[Start] Network weight sum:',agent.w:sum())

-- ``steps`` is the number of training steps to perform
-- There are not separate loops that loop over, for example,
-- different epochs, trajectories, etc. Instead, the we just keep iterating over
-- steps, and logic within the loop handles moving to the next epoch, moving to the
-- next trajectory, when to start testing, etc.
while step < opt.steps do
    step = step + 1
    if not RANDOM_TEST then
        -- Not testing the random policy

        xlua.progress(step, opt.steps)

        -- call to the Deep Q-Learner
        local action_index, object_index = agent:perceive(
            reward, state, terminal, nil, nil, available_objects, priority)

        if reward > 0 then
            pos_reward_cnt = pos_reward_cnt + 1
        end

        -- game over? get next game!
        if not terminal then
            -- step the environment
            state, reward, terminal, available_objects = framework.step(action_index, object_index)

            -- priority sweeping for positive rewards
            if reward > 0 then
                priority = true
            else
                priority = false
            end
        else
            state, reward, terminal, available_objects = framework.newGame()
        end

        if step % opt.prog_freq == 0 then
            assert(step==agent.numSteps, 'trainer step: ' .. step ..
                    ' & agent.numSteps: ' .. agent.numSteps)
            print("\nSteps: ", step, " | Achieved quest level, current reward:" , pos_reward_cnt)
            agent:report()
            pos_reward_cnt = 0
        end

        if step%1000 == 0 then
            collectgarbage()
        end
    end

    -- Testing
    if step % opt.eval_freq == 0 and step > learn_start then
        print('Testing Starts ... ')
        quest3_reward_cnt = 0
        quest2_reward_cnt = 0
        quest1_reward_cnt = 0
        test_avg_Q = test_avg_Q or optim.Logger(paths.concat(opt.exp_folder , 'test_avgQ.log'))
        test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))
        test_quest1 = test_quest1 or optim.Logger(paths.concat(opt.exp_folder , 'test_quest1.log'))
        if TUTORIAL_WORLD then
            test_quest2 = test_quest2 or optim.Logger(paths.concat(opt.exp_folder , 'test_quest2.log'))
            test_quest3 = test_quest3 or optim.Logger(paths.concat(opt.exp_folder , 'test_quest3.log'))
        end

        gameLogger = gameLogger or io.open(paths.concat(opt.exp_folder, 'game.log'), 'w')

        state, reward, terminal, available_objects = framework.newGame(gameLogger)

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            xlua.progress(estep, opt.eval_steps)

            local action_index, object_index, q_func
            if not RANDOM_TEST then
                action_index, object_index, q_func = agent:perceive(reward, state, terminal, true, 0.05, available_objects)
            else
                action_index, object_index, q_func = agent:perceive(reward, state, terminal, true, 1, available_objects)
            end

             -- print Q function for previous state
            if q_func then
                local actions = framework.getActions()
                local objects = framework.getObjects()
                for i=1, #actions do
                    gameLogger:write(actions[i],' ', q_func[1][i],'\n')
                end
                gameLogger:write("-----\n")
                for i=1, #objects do
                    gameLogger:write(objects[i],' ', q_func[2][i], '\n')
                end

            else
                gameLogger:write("Random action\n")
            end

            -- Play game in test mode (episodes don't end when losing a life)
            state, reward, terminal, available_objects = framework.step(action_index, object_index, gameLogger)

            if TUTORIAL_WORLD then
                if(reward > 9) then
                    quest1_reward_cnt =quest1_reward_cnt+1
                elseif reward > 0.9 then
                    quest2_reward_cnt = quest2_reward_cnt + 1
                elseif reward > 0 then
                    quest3_reward_cnt = quest3_reward_cnt + 1 --defeat guardian
                end
            else
                if(reward > 0.9) then
                    quest1_reward_cnt =quest1_reward_cnt+1
                end
            end

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                state, reward, terminal, available_objects = framework.newGame(gameLogger)
            end
        end -- for

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time

        if not RANDOM_TEST then
            agent:compute_validation_statistics()
        end

        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print(
            "V", v_history[ind], "TD error",
            td_history[ind], "V avg:", v_history[ind])

        --saving and plotting
        test_avg_R:add{['% Average Reward'] = total_reward}
        test_avg_Q:add{['% Average Q'] = agent.v_avg}
        test_quest1:add{['% Quest 1'] = quest1_reward_cnt/nepisodes}
        if TUTORIAL_WORLD then
            test_quest2:add{['% Quest 2'] = quest2_reward_cnt/nepisodes}
            test_quest3:add{['% Quest 3'] = quest3_reward_cnt/nepisodes}
        end

        test_avg_R:style{['% Average Reward'] = '-'}; test_avg_R:plot()
        test_avg_Q:style{['% Average Q'] = '-'}; test_avg_Q:plot()
        test_quest1:style{['% Quest 1'] = '-'}; test_quest1:plot()
        if TUTORIAL_WORLD then
            test_quest2:style{['% Quest 2'] = '-'}; test_quest2:plot()
            test_quest3:style{['% Quest 3'] = '-'}; test_quest3:plot()
        end

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d, completion rate: %.2f',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards, pos_reward_cnt/nepisodes))


        pos_reward_cnt = 0
        quest1_reward_cnt = 0
        gameLogger:write("###############\n\n") -- end of testing epoch
        print('Testing Ends ... ')
        collectgarbage()
    end -- Testing

    -- Saving
    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            print('Network weight sum:', w:sum())
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end

        if EMBEDDING then
            -- save word embeddings
            embedding_mat = EMBEDDING:forward(torch.range(1, #symbols+1))
            embedding_save = {}
            for i=1, embedding_mat:size(1)-1 do
                embedding_save[symbols[i]] = embedding_mat[i]
            end
            embedding_save["NULL"] = embedding_mat[embedding_mat:size(1)]

            -- description embeddings
            local desc_embeddings
            if ANALYZE_TEST then
                require 'descriptions'
                desc_embeddings = {}
                for i=1, #DESCRIPTIONS do
                    local embeddings = {}
                    for j=1, #DESCRIPTIONS[i] do
                        local input_vec = framework.vector_function(DESCRIPTIONS[i][j])
                        local state_tmp = tensor_to_table(input_vec, self.state_dim, self.hist_len)
                        local output_vec = LSTM_MODEL:forward(state_tmp)
                        table.insert(embeddings, output_vec)
                    end
                    table.insert(desc_embeddings, embeddings)
                end
            end
            torch.save(
                filename..'.embeddings.t7',
                {embeddings = embedding_save, symbols=symbols, desc_embeddings=desc_embeddings})
        end

        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()

        if ANALYZE_TEST then
            return
        end
    end -- Saving
end
