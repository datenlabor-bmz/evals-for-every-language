import React from 'react';
import { Box, Typography, Tab, Tabs, IconButton } from '@mui/material';
import DarkModeOutlinedIcon from '@mui/icons-material/DarkModeOutlined';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';

function Header() {
  const [value, setValue] = React.useState(0);

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  return (
    <Box sx={{ textAlign: 'center', mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <div style={{ fontSize: '50px' }}>🤗</div>
      </Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Open LLM Leaderboard Archived
      </Typography>
      <Typography variant="subtitle1" gutterBottom sx={{ color: 'text.secondary' }}>
        Comparing Large Language Models in an <strong>open</strong> and <strong>reproducible</strong> way
      </Typography>
    
    </Box>
  );
}

export default Header;