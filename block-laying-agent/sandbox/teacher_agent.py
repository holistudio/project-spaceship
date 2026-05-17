teacher_config = {
    "target_object_id": 0, 
    "num_user_blocks": 3, 
    "num_agent_blocks": 1, 
    "target_completion": 0.98, 
    "reward_function":{
        "unmoved_score": 1,
        "moved_score": -0.1,
        "removed_score": -2,
        "impatience": 0.0,
    }
    
}

# class TeacherAgent
# __init__(config):
    # num_user_blocks = num_user_blocks
    # num_agent_blocks = num_agent_blocks
    # target_object, total_target_voxels = _load_target_voxel(target_object_id)
    # target_completion = target_completion

    # steps = 0
    # impatience = impatience
    # unmoved_score = 1
    # moved_score = -0.1
    # removed_score = -2

# _load_target_voxel(target_object_id):
    # return target_object, total_target_voxels

# place_block(voxel_grid):
    # figure out where to put the next block
    # TODO: for now use only voxel grid to decide next block
    # later, consider using only user_blocks for the "first half" of episode to decide (epsilon decay?)
    # block = block_type_i, orientation, x, y, z
    
    # update voxel grid

    # return block, voxel_grid

# check_terminal(voxel_grid):
    # compare voxel grid with target object
    # compute cells that correctly overlap
    # perc_complete = num_correct / total_target_voxels
    # if perc_complete > target_completion:
        # return True
    # else:
        # return False

# _remove_block(block, voxel_grid):
    # return voxel_grid

# _move_block(block, new_position, new_orientation, voxel_grid):
    # _remove_block(block, voxel_grid)
    # add block to voxel_grid new_position, new_orientation
    # return voxel_grid

# change_agent_blocks(agent_blocks):
    # latest_blocks = agent_blocks[-num_agent_blocks:]
    # agent_block_changes = [[block.id, "unchanged"] for block in latest_blocks]

    # for each of the latest agent blocks, change accordingly
    # for i,block in enumerate(latest_blocks):
        # if the block does not intersect 
        # the target object filled volume, it is REMOVED
        # ...
        # voxel_grid = _remove_block(block, voxel_grid)
        # agent_block_changes[i][1] = "removed"

        # if the block is within the target object volume, 
        # but > TARGET_DISTANCE, the block is MOVED 
        # to a location closer to the other user_blocks
        # ...
        # voxel_grid = _move_block(block, new_position, new_orientation, voxel_grid)
        # agent_block_changes[i][1] = "moved"
    
    # return agent_block_changes, voxel_grid

# reward_function(agent_block_changes):
    # score = steps * impatience
    # for _, change in agent_block_changes:
        # if change == "unmoved": score += unmoved_score
        # if change == "moved": score += moved_score
        # if change == "removed": score += removed_score
    
    # return score

# step(observation):
    # voxel_grid = observation["voxel_grid"]

    # check if the target object is mostly complete
    # done = check_terminal(voxel_grid)

    # if done:
        # return {"done": True}
    # else:
        # actions = {
        #     "new_blocks": [],
        #     "change_blocks": [],
        #     "score": 0.0, # TODO: score assumed only for evaluating latest block (not a cumulative reward)
        #     "done": False,
        # }

        # agent_blocks = observation["agent_blocks"]
        # if len(agent_blocks) > 0:
            # change latest blocks placed by agents
            # agent_block_changes, voxel_grid = change_agent_blocks(agent_blocks)
            # actions["score"] = reward_function(agent_block_changes)

        # place new blocks
        # for _ in range(num_user_blocks):
            # block, voxel_grid = place_block(voxel_grid)
            # actions["new_blocks"].append(block)

        # return actions
