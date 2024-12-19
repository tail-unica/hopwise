def pytest_collection_modifyitems(session, config, items):
    items_names = [item.name for item in items]
    cmd_line_test_index = items_names.index("test_config_command_line")
    items.insert(len(items), items.pop(cmd_line_test_index))
