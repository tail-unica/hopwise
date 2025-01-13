def pytest_collection_modifyitems(session, config, items):
    cmd_test_name = "test_config_command_line"
    items_names = [item.name for item in items]
    if cmd_test_name in items_names:
        cmd_line_test_index = items_names.index(cmd_test_name)
        items.insert(len(items), items.pop(cmd_line_test_index))
